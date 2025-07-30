import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import numpy as np
import cv2
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

model = YOLO("yolov8n.pt")

def bbox_to_kalman_state(bbox):
    # 바운딩 박스를 칼만 필터 상태로 변환
    u = bbox[0] + (bbox[2] - bbox[0]) / 2
    v = bbox[1] + (bbox[3] - bbox[1]) / 2
    s = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    r = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
    
    return np.array([u, v, s, r, 0, 0, 0])

def kalman_state_to_bbox(state):
    # 칼만 필터 상태를 바운딩 박스로 변환
    u, v, s, r = state[0], state[1], state[2], state[3]

    #디버깅용
    s_r = s * r
    if not np.isfinite(s_r):
        raise ValueError(f"[ERROR] s * r is not finite! Got s={s}, r={r}, s*r={s_r}")
    if s_r < 0:
        raise ValueError(f"[ERROR] s * r is negative! Got s={s}, r={r}, s*r={s_r}")
    
    w = np.sqrt(s * r)
    h = s / w
    x1 = u - w / 2
    y1 = v - h / 2
    x2 = u + w / 2
    y2 = v + h / 2

    return [x1, y1, x2, y2]

def cosine_similarity(v1,v2):
    v1 = np.array(v1) # v1 = v_tarck  (2프레임전 관측값 - 1프레임전 관측값)
    v2 = np.array(v2) # v2 = v_candidate (1프레임전 관측값 - 현재관측값)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1==0 or norm2==0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)

# ===================== 칼만 필터 클래스 =====================
class KalmanBox:
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4) # 칼만 필터 초기화
        self.kf.x = bbox_to_kalman_state(bbox) # 칼만 필터 상태 초기화
        self.kf.P *= 1000. # 초기 오차 공분산 행렬
        self.kf.F = np.eye(7) # 상태 전이 행렬
        for i in range(4, 7):
            self.kf.F[i - 4, i] = 1 # 칼만 필터 상태 전이 행렬
        self.kf.H = np.zeros((4, 7)) # 관측 모델 행렬
        self.kf.H[0, 0] = 1 # 칼만 필터 상태에서 x 좌표
        self.kf.H[1, 1] = 1 # 칼만 필터 상태에서 y 좌표
        self.kf.H[2, 2] = 1 # 칼만 필터 상태에서 너비
        self.kf.H[3, 3] = 1 # 칼만 필터 상태에서 높이
        self.kf.R *= 10 # 관측 잡음 공분산 행렬
        self.kf.Q *= 0.01 # 프로세스 잡음 공분산 행렬
    
    def predict(self):
        # 칼만 필터 예측
        s_before = self.kf.x[2]
        ds_before = self.kf.x[6]

        self.kf.predict()
        
        #디버깅용
        # 예측 상태에서 s, r 추출
        s_after = self.kf.x[2]
        r = self.kf.x[3]
        s_r = s_after * r
        # NaN, inf, 음수 체크 → 오류 발생시 중단
        if not np.isfinite(s_after):
            raise ValueError(f"[PREDICT ERROR] s is not finite: s={s_after}")
        if not np.isfinite(r):
            raise ValueError(f"[PREDICT ERROR] r is not finite: r={r}")
        if not np.isfinite(s_r):
            raise ValueError(f"[PREDICT ERROR] s*r is not finite: s={s_after}, r={r}, s*r={s_r}")
        if s_after <= 0:
            raise ValueError(f"[PREDICT ERROR] (s_predicted={s_after}) | (s_before={s_before}, ds={ds_before})")
        if r <= 0:
            raise ValueError(f"[PREDICT ERROR] r is non-positive: r={r}")
        if s_r <= 0:
            raise ValueError(f"[PREDICT ERROR] s*r is non-positive: s*r={s_r}")
        
        return kalman_state_to_bbox(self.kf.x)
    
    def update(self, bbox):
        # 칼만 필터 업데이트
        z = bbox_to_kalman_state(bbox)
        z = np.atleast_2d(z[:4]).T
        self.kf.update(z)
        return kalman_state_to_bbox(self.kf.x)

# ===================== OCSORT 클래스 =====================
class OCSort:
    def __init__(self):
        # SORT 알고리즘 초기화
        # 추적중인 객체 리스트와 추적 ID 초기화
        self.tracked_objects = []
        self.track_id = 0
        self.z_t_minus_2 = None
        self.z_t_minus_1 = None

    def detection(self, frame):
        return model(frame)

    def extract_detections(self, frame):
        # car와 truck 객체를 필터링 (2: car, 3: truck)
        detections = []
        results = self.detection(frame)
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) in [2,3]: 
                    x1, y1, x2, y2 = box.xyxy[0]
                    detections.append({
                        'bbox': [x1.item(), y1.item(), x2.item(), y2.item()],
                        'confidence': box.conf[0].item(),
                        'class': int(box.cls[0]),
                    })
        return detections

    def iou(self, box1, box2):
        # 두 박스의 IoU 계산(교집합 영역의 비율)
        xA = max(box1[0], box2[0]) # 교차 영역의 왼쪽 x 좌표
        yA = max(box1[1], box2[1]) # 교차 영역의 위쪽 y 좌표
        xB = min(box1[2], box2[2]) # 교차 영역의 오른쪽 x 좌표
        yB = min(box1[3], box2[3]) # 교차 영역의 아래쪽 y 좌표

        inter = max(0, xB - xA) * max(0, yB - yA) # 교집합 영역
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / float(area1 + area2 - inter + 1e-6) # 0으로 나누는 오류 방지

    def match(self, detections, trackers, iou_threshold=0.3,ocm_threhold = 0.3):

        if len(trackers) == 0: # 트래커가 없으면 매칭할 필요 없음
            return [], [], list(range(len(detections))) 

        cost_iou = np.zeros((len(trackers), len(detections))) 
        cost_ocm = np.zeros((len(trackers), len(detections)))  
        cost_matrix = np.zeros((len(trackers), len(detections))) 

        for i, trk in enumerate(trackers):
            if trk['z_t_minus_2']==None:
                for j, det in enumerate(detections):
                    cost_iou[i, j] = 1 - self.iou(trk['bbox'], det['bbox']) 
                    cost_ocm[i, j] = 0
                    cost_matrix[i, j] = cost_iou[i,j] + ocm_threhold*cost_ocm[i,j]
            else:
                v1 = np.array(trk['z_t_minus_1']) - np.array(trk['z_t_minus_2'])
                for j, det in enumerate(detections):
                    v2 = np.array(det['bbox']) - np.array(trk['z_t_minus_1'])
                    cost_iou[i, j] = 1 - self.iou(trk['bbox'], det['bbox']) 
                    cost_ocm[i, j] = 1 - cosine_similarity(v1,v2)
                    cost_matrix[i, j] = cost_iou[i,j] + ocm_threhold*cost_ocm[i,j] # iou와 ocm cost 가중치 적용한 cost_matrix

        row_ind, col_ind = linear_sum_assignment(cost_matrix) # 헝가리안 알고리즘으로 최적 매칭

        # 매칭된 트래커와 detection을 저장
        matches = []
        unmatched_trackers = list(range(len(trackers)))
        unmatched_detections = list(range(len(detections))) 

        # 매칭된 트래커와 detection을 IoU 임계값에 따라 필터링
        for r, c in zip(row_ind, col_ind):
            if 1 - cost_matrix[r, c] > iou_threshold:
                matches.append((r, c))
                unmatched_trackers.remove(r)
                unmatched_detections.remove(c)

        # 매칭된 트래커와 detection을 반환
        return matches, unmatched_trackers, unmatched_detections

    def update(self, detections, frame_num):
        '''
            핵심 로직
            1. 기존 트래커들 predict()
            2. 매칭된 트래커 업데이트
            3. re-id로 매칭된 tracker의 가상 궤적 보정 (OCR)
            4. 매칭 안 된 detection -> 새로운 트래커 생성, or re-id인지 확인
            5. 매칭 안 된 tracker 삭제
        '''
        # 1. 기존 트래커 예측
        for trk in self.tracked_objects: # tracked_objects는 [{'id': id, 'kf': KalmanBox, 'bbox': [x1,y1,x2,y2]}, ...] 형태
            try:
                trk['bbox'] = trk['kf'].predict()
                trk['age'] += 1
            except ValueError as e:
                print("\n====[TRACKING PREDICT ERROR]====")
                print(f"class: {trk['class']}")
                print(f"Tracker ID: {trk['id']}")
                print(f"bbox: {trk.get('bbox')}")
                print(f"age: {trk.get('age')}")
                print(f"time_since_update: {trk.get('time_since_update')}")
                print(f"start_missing_frame_num: {trk.get('start_missing_frame_num')}")
                print(f"[Kalman State] predict_s={trk['kf'].kf.x[2]}, ds={trk['kf'].kf.x[6]}")

        matches, unmatched_trks, unmatched_dets = self.match(detections, self.tracked_objects)

        # 2. 매칭된 트래커 업데이트
        re_id = [] # 매칭 후 re-id 선별 전 re-id 초기화
        for t, d in matches:
            if self.tracked_objects[t]['time_since_update'] >=1: # re-id의 첫 번째 조건
                re_id.append((t,d))
            else: 
                self.tracked_objects[t]['time_since_update'] = 0
                self.tracked_objects[t]['z_t_minus_2'] = self.tracked_objects[t]['z_t_minus_1']
                self.tracked_objects[t]['z_t_minus_1'] = detections[d]['bbox']
                self.tracked_objects[t]['bbox'] = self.tracked_objects[t]['kf'].update(detections[d]['bbox'])

         # 3. re-id로 매칭된 tracker의 가상 궤적 보정 (OCR)
        for t,d in re_id:
            trk = self.tracked_objects[t]
            z_start = trk['z_t_minus_1'] # re-id 되기 전의 마지막 tracking 값
            z_end = detections[d]['bbox'] # re-id 된 tracking 값
            num_missing_frame = frame_num - trk['start_missing_frame_num'] + 1 # 객체가 사라진 frame수 

            virtual_obs_list = []
            for i in range(1,num_missing_frame+1):
                alpha = i / (num_missing_frame+1)
                z_virtual = (1 - alpha) * np.array(z_start) + alpha * np.array(z_end)
                virtual_obs_list.append(z_virtual)
            for z_virtual in virtual_obs_list:
                trk['kf'].predict()
                trk['bbox'] = trk['kf'].update(z_virtual)

            # 마지막 진짜 detection으로 업데이트
            trk['kf'].predict()
            trk['bbox'] = trk['kf'].update(z_end)
            trk['time_since_update'] = 0
            trk['z_t_minus_2'] = trk['z_t_minus_1']
            trk['z_t_minus_1'] = z_end
            trk['start_missing_frame_num'] = 're-identified!'

        # 4. 매칭 안 된 detection -> 새로운 트래커 생성
        for d in unmatched_dets:
            self.track_id += 1
            kf = KalmanBox(detections[d]['bbox'])
            self.tracked_objects.append({
                'id': self.track_id,
                'kf': kf,
                'bbox': detections[d]['bbox'],
                'time_since_update': 0,
                'age':1,
                'class': detections[d]['class'],
                'z_t_minus_1' : detections[d]['bbox'],
                'z_t_minus_2' : self.z_t_minus_1,
                'start_missing_frame_num': 'not-yet'
            })    

        # 5. 매칭 안 된 tracker 삭제
        max_age = 3
        for t in sorted(unmatched_trks, reverse=True):
            self.tracked_objects[t]['time_since_update'] += 1
            self.tracked_objects[t]['start_missing_frame_num'] = frame_num
            
            if self.tracked_objects[t]['time_since_update'] >= max_age:
                del self.tracked_objects[t] # 아예 관측되었던 object에서 삭제

        return self.tracked_objects

# ===================== 메인 =====================
if __name__ == "__main__":
    ocsort = OCSort()
    cap = cv2.VideoCapture("dongwon_building-13.avi")

    total_frames = 0
    total_time = 0.0
    total_latency = 0.0
    extract_time = 0.0
    update_time = 0.0

    
    frame_num = 0 

    start_time = time.time()
    while cap.isOpened():
        success, frame = cap.read()
        
        if frame is None:
            print("No frame to process.")
            break       

        if not success:
            print("Failed to read frame from video.")
            break
        total_frames += 1
        frame_start = time.time()

        t1 = time.time()
        detections = ocsort.extract_detections(frame)
        t2 = time.time()
        extract_time += (t2 - t1)

        t3 = time.time()
        tracked = ocsort.update(detections,frame_num)
        t4 = time.time()
        update_time += (t4 - t3)

        frame_end = time.time()
        frame_latency = frame_end - frame_start
        total_latency += frame_latency

        # 추적 결과 시각화
        for trk in tracked:
            x1, y1, x2, y2 = [int(v) for v in trk['bbox']]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {trk['id']}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("OCSORT + YOLOv8 Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_num += 1
    
    end_time = time.time()
    total_time = end_time - start_time
    cap.release()
    cv2.destroyAllWindows()

    # ===================== 결과 출력 =====================
    print(f"\n========== Performance Metrics ==========")
    print(f"Total frames processed     : {total_frames}")
    print(f"Total elapsed time         : {total_time:.2f} sec")
    print(f"FPS                        : {total_frames / total_time:.2f}")
    print(f"Avg latency per frame      : {total_latency / total_frames:.4f} sec")

    print(f"Feature extraction time    : {extract_time:.2f} sec "
          f"({(extract_time / total_time * 100):.1f}%)")
    
    print(f"Tracker update time        : {update_time:.2f} sec "
          f"({(update_time / total_time * 100):.1f}%)")

    other_overhead = total_time - (extract_time + update_time)
    print(f"Other overhead (I/O, vis)  : {other_overhead:.2f} sec "
          f"({(other_overhead / total_time * 100):.1f}%)")

    print(f"=========================================")