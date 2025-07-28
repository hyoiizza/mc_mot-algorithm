import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import cv2
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

model = YOLO("yolov8n.pt")

#===================== 헬퍼함수 ==============================
def bbox_to_kalman_state(bbox):
    # 바운딩 박스를 칼만 필터 상태로 변환
    u = bbox[0] + (bbox[2] - bbox[0]) / 2
    v = bbox[1] + (bbox[3] - bbox[1]) / 2
    g = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1]) # 종횡비
    h =  bbox[3] - bbox[1] # 높이
    
    return np.array([u, v, g, h, 0, 0, 0, 0])

def kalman_state_to_bbox(state):
    # 칼만 필터 상태를 바운딩 박스로 변환
    u, v, g, h = state[0], state[1], state[2], state[3]
    
    w = g*h

    x1 = u - w / 2
    y1 = v - h / 2
    x2 = u + w / 2
    y2 = v + h / 2

    return [x1, y1, x2, y2]

def cosine_similarity(v1,v2):
    v1 = np.array(v1) 
    v2 = np.array(v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1==0 or norm2==0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)


# ===================== 칼만 필터 클래스 =====================
class KalmanBox:
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=8, dim_z=4) # 칼만 필터 초기화
        self.kf.x = bbox_to_kalman_state(bbox) # 칼만 필터 상태 초기화
        self.kf.P *= 1000. # 초기 오차 공분산 행렬
        self.kf.F = np.eye(8) # 상태 전이 행렬
        for i in range(4):
            self.kf.F[i, i+4] = 1 # 칼만 필터 상태 전이 행렬
        self.kf.H = np.zeros((4, 8)) # 관측 모델 행렬
        self.kf.H[0, 0] = 1 # u
        self.kf.H[1, 1] = 1 # v
        self.kf.H[2, 2] = 1 # gamma
        self.kf.H[3, 3] = 1 # h
        # 공분산 설정
        self.kf.R *= 10
        self.kf.Q *= 0.01 
    
    def predict(self):
        # 칼만 필터 예측
        self.kf.predict()
        #디버깅용
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


# ===================== SORT 클래스 =====================
class Deepsort:
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
                if int(box.cls[0]) in [2, 3]: 
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
    
    def mahalanobis_distance(self, trk_kf, detection_vec):
        """
        trk_kf: KalmanFilter 객체 (filterpy.kalman.KalmanFilter)
        detection_vec: np.array([cx, cy, aspect_ratio, h]) shape=(4,)
        """
        # 예측 관측값 (4x1)
        y = np.dot(trk_kf.H, trk_kf.x)  # shape (4, 1)
        y = y.flatten()                 # shape (4,)

        # 예측 관측 공분산 행렬 S = HPH^T + R
        S = np.dot(np.dot(trk_kf.H, trk_kf.P), trk_kf.H.T) + trk_kf.R  # shape (4, 4)
        S_inv = np.linalg.inv(S)

        # 거리 계산
        diff = detection_vec - y
        dist = np.dot(np.dot(diff.T, S_inv), diff)
        return dist

    def ds_cnn(self, bbox, frame):
        x1, y1, x2, y2 = bbox
        img_crop = frame[y1:y2, x1:x2]

        input = cv2.resize(img_crop, (128,256))
        


    def match(self, frame, detections, trackers, iou_threshold=0.3,alpha=0.5,maha_threshold=9.4877 ,appearance_threshold=0):
        if len(trackers) == 0: # 트래커가 없으면 매칭할 필요 없음
            return [], [], list(range(len(detections))) 

        cost_matrix = np.zeros((len(trackers), len(detections))) 
        maha_dist = np.zeros((len(trackers), len(detections))) 
        appearance = np.zeros((len(trackers), len(detections)))
        appear_gate = np.zeros((len(trackers), len(detections)))
        fin_gate = np.zeros((len(trackers), len(detections)))
        maha_dist_gate = np.zeros((len(trackers), len(detections)))\
        
        for i, trk_idx in enumerate(trackers):
            trk = self.tracked_objects[trk_idx]
            kf = trk['kf']  # KalmanBox 객체
            for j, det_idx in enumerate(detections):
                det_bbox = detections[det_idx]['bbox']  # [x1, y1, x2, y2]
                det_vec = bbox_to_kalman_state(det_bbox)[:4]  # [cx, cy, aspect, h]

                # Mahalanobis distance 계산
                maha = self.mahalanobis_distance(kf, det_vec)
                maha_dist[i, j] = maha
                if maha_dist[i,j] > maha_threshold: # 9.4877 = 95% 신뢰수준
                    maha_dist_gate[i,j] = 1
                else 
                    maha_dist_gate[i,j] = 0

                # cnn기반 외형정보 오차 계산
                feature1 = self.ds_cnn(trk['bobx'])
                feature2 = self.ds_cnn(det_bbox)
                appearance[i,j] = 1 - cosine_similarity(feature1, feature2)
                if appearance[i,j] > appearance_threshold:
                    appear_gate[i,j] = 1
                else 
                    appear_gate[i,j] = 0
                # 최종 cost 가중치 조합
                fin_gate[i,j] = appear_gate[i,j] * maha_dist_gate[i,j]
                cost_matrix[i, j] = fin_gate[i,j]{alpha*appearance[i,j] + (1-alpha)*maha_dist[i,j]}
        

        # time since update을 기준으로 우선 배정..
        row_ind, col_ind = linear_sum_assignment(cost_matrix) # 헝가리안 알고리즘으로 최적 매칭

        # 매칭된 트래커와 detection을 저장
        matches = []
        unmatched_trackers = list(range(len(trackers)))
        unmatched_detections = list(range(len(detections)))

        # 매칭된 트래커와 detection을 IoU 임계값에 따라 필터링 , # time_since_update에 따라 우선 배정 로직 추가 필요
        for r, c in zip(row_ind, col_ind):
            if 1 - cost_matrix[r, c] > iou_threshold:
                matches.append((r, c))
                unmatched_trackers.remove(r)
                unmatched_detections.remove(c)

        # 매칭된 트래커와 detection을 반환
        return matches, unmatched_trackers, unmatched_detections

    def update(self, frame,detections,iou_threshold=0.3):
        '''
            핵심 로직
            1. 기존 트래커들 predict()
            2. 매칭된 트래커 업데이트
            3. 매칭 안 된 detection, trackers에 대해 두번째 매칭 (iou기반)
            4. 두번째 매칭에 의해 매칭된 트래커 업데이트 
            5. 두번째 매칭에 의해 새로 발견한 객체 업데이트(new track)
            6. 트래커 삭제
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
                print(f"[Kalman State] predict_s={trk['kf'].kf.x[2]}, ds={trk['kf'].kf.x[6]}")

        matches, unmatched_trks, unmatched_dets = self.match(frame, detections, self.tracked_objects)

        # 2. 매칭된 트래커 업데이트
        for t, d in matches:
            if self.tracked_objects[t]['state'] == 'confirmed':
                self.tracked_objects[t]['time_since_update'] = 0
                self.tracked_objects[t]['bbox'] = self.tracked_objects[t]['kf'].update(detections[d]['bbox'])
                self.tracked_objects[t]['hits'] +=1

            if self.tracked_objects[t]['state'] == 'tentative':
                self.tracked_objects[t]['hits'] +=1
                self.tracked_objects[t]['time_since_update'] = 0
                self.tracked_objects[t]['bbox'] = self.tracked_objects[t]['kf'].update(detections[d]['bbox'])
                
                if self.tracked_objects[t]['hits'] >=3:
                    self.tracked_objects[t]['state'] == 'confirmed'

        # 3. 매칭 안 된 detection, trackers에 대해 두번째 매칭 (iou기반)
        cost_matrix_2 = np.zeros((len(unmatched_trks), len(unmatched_dets))) 
        for i, d in enumerate(unmatched_dets):
            for j, t in enumerate(unmatched_trks):
                trk2 = self.tracked_objects[unmatched_trks[j]] # ========================
                det2 = detections[unmatched_dets[i]] #=============================
                cost_matrix_2[i, j] = 1 - self.iou(trk2['bbox'], det2['bbox']) 

        row_ind, col_ind = linear_sum_assignment(cost_matrix_2)
        
        matches_2 = []
        unmatched_trackers_2 = list(range(len(unmatched_trks)))
        unmatched_detections_2 = list(range(len(unmatched_dets)))

        # 매칭된 트래커와 detection을 IoU 임계값에 따라 필터링
        for r, c in zip(row_ind, col_ind):
            if 1 - cost_matrix_2[r, c] > iou_threshold:
                matches_2.append((r, c))
                unmatched_trackers_2.remove(r)
                unmatched_detections_2.remove(c)
        # 4. 두번째 매칭에 의해 매칭된 트래커 업데이트 
        for t,d in matches_2:
            if self.tracked_objects[t]['state'] =='tentative':
                self.tracked_objects[t]['hits'] +=1
                self.tracked_objects[t]['time_since_update'] = 0
                self.tracked_objects[t]['bbox'] = self.tracked_objects[t]['kf'].update(detections[d]['bbox'])
        # 5. 두번째 매칭에 의해 새로 발견한 객체 업데이트(new track)
        for d in unmatched_detections_2:
            self.track_id += 1
            kf = KalmanBox(detections[d]['bbox'])
            self.tracked_objects.append({
                'id': self.track_id,
                'kf': kf,
                'bbox': detections[d]['bbox'],
                'time_since_update': 0,
                'age':1,
                'class': detections[d]['class'],
                'start_missing_frame_num': 'not-yet',
                'state' : 'tentative',
                'hits' : 1
            }) 
        # 6. 트래커 삭제 
        max_age = 3
        for t in sorted(unmatched_trackers_2, reverse=True):
            self.tracked_objects[t]['time_since_update'] += 1
            #tentative 상태에서 삭제
            if self.tracked_objects[t]['state']=='tentatvie' and self.tracked_objects[t]['time_since_update'] >= 3:
                del self.tracked_objects[t] # 아예 관측되었던 object에서 삭제
            #confirm 상태에서 삭제
            elif self.tracked_objects[t]['state'] == 'confirmed' and self.tracked_objects[t]['time_since_update'] >= max_age:
                del self.tracked_objects[t]

        return self.tracked_objects


# ===================== 메인 =====================
if __name__ == "__main__":
    deepsort = Deepsort()
    cap = cv2.VideoCapture("dongwon_building-09.avi")

    frame_num = 0 

    while cap.isOpened():
        success, frame = cap.read()
        
        if frame is None:
            print("No frame to process.")
            break       

        if not success:
            print("Failed to read frame from video.")
            break
        
    
        detections = deepsort.extract_detections(frame) 
        tracked = deepsort.update(frame, detections)

        # 추적 결과 시각화
        for trk in tracked:
            x1, y1, x2, y2 = [int(v) for v in trk['bbox']]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {trk['id']}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Deepsort + YOLOv8 Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_num += 1
        print('현재',frame_num,'번째 frame')

    cap.release()
    cv2.destroyAllWindows()
