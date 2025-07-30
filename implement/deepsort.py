import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from encoder_model  import DeepSortEncoder
import time


#==================== 필요한 모델 정의 =========================
model = YOLO("yolov8n.pt")

encoder = DeepSortEncoder()

file_dir = os.path.dirname(__file__)
weight_path = os.path.join(file_dir, "deepsort_encoder.pth")

encoder.load_state_dict(torch.load(weight_path, map_location='cpu'))  # 또는 'cuda'
encoder.eval()

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
    # 두 벡터간의 코사인 유사도
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
        '''
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
        '''
        return kalman_state_to_bbox(self.kf.x)
    
    def update(self, bbox):
        # 칼만 필터 업데이트
        z = bbox_to_kalman_state(bbox)
        z = np.atleast_2d(z[:4]).T
        self.kf.update(z)
        return kalman_state_to_bbox(self.kf.x)

#====================== appearnce extractor: encoder =======
transform = transforms.Compose([
    transforms.Resize((128, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def extract_feature_from_crop(bbox_crop_np, encoder_model, device='cpu'):
    # BGR(OpenCV) to RGB(PIL)
    if isinstance(bbox_crop_np, np.ndarray):
        image = Image.fromarray(bbox_crop_np[..., ::-1])
    else:
        image = bbox_crop_np

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        feature = encoder_model(input_tensor)  # shape: (1, 128)

    return feature.squeeze(0).cpu().numpy()

# ===================== Deep SORT 클래스 ====================
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
                        'feature' : np.zeros(128)
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
        kf = trk_kf.kf

        # 예측 관측값 (4x1)
        y = np.dot(kf.H, kf.x)  # shape (4, 1)
        y = y.flatten()                 # shape (4,)

        # 예측 관측 공분산 행렬 S = HPH^T + R
        S = np.dot(np.dot(kf.H, kf.P), kf.H.T) + kf.R  # shape (4, 4)
        S_inv = np.linalg.inv(S)

        # 거리 계산
        diff = detection_vec - y
        dist = np.dot(np.dot(diff.T, S_inv), diff)
        return dist

    def match(self, frame, detections, trackers, alpha=0.5,maha_threshold=9.4877 ,appearance_threshold=0):
        if len(trackers) == 0: # 트래커가 없으면 매칭할 필요 없음
            return [], [], list(range(len(detections))) 

        trackers = list(range(len(self.tracked_objects)))

        cost_matrix = np.zeros((len(trackers), len(detections))) 
        maha_dist = np.zeros((len(trackers), len(detections))) 
        appearance = np.zeros((len(trackers), len(detections)))
        appear_gate = np.zeros((len(trackers), len(detections)))
        fin_gate = np.zeros((len(trackers), len(detections)))
        maha_dist_gate = np.zeros((len(trackers), len(detections)))\
        
        # tracker feature 저장
        tracker_features = []
        for i, trk_idx in enumerate(trackers):
            trk = self.tracked_objects[trk_idx]
            bbox = trk['bbox']
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            crop = frame[y1:y2, x1:x2] 
            feature = extract_feature_from_crop(crop, encoder, device='cpu')
            tracker_features.append(feature)
            
        
        # detection feature 저장
        detection_features = []
        for j, det in enumerate(detections):
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[y1:y2, x1:x2]  
            feature = extract_feature_from_crop(crop, encoder, device='cpu')
            detection_features.append(feature)
            det['feature'] = feature


        for i, trk_idx in enumerate(trackers):
            trk = self.tracked_objects[trk_idx]
            kf = trk['kf']  # KalmanBox 객체
            for j, det in enumerate(detections):
                det_bbox = det['bbox']  # [x1, y1, x2, y2]
                det_vec = bbox_to_kalman_state(det_bbox)[:4]  # [cx, cy, aspect, h]

                # Mahalanobis distance 계산
                maha = self.mahalanobis_distance(kf, det_vec)
                maha_dist[i, j] = maha
                if maha_dist[i,j] > maha_threshold: # 9.4877 = 95% 신뢰수준
                    maha_dist_gate[i,j] = 1
                else:
                    maha_dist_gate[i,j] = 0

                # cnn기반 외형정보 오차 계산
                appearance[i,j] = 1 - cosine_similarity(detection_features[j], tracker_features[i])
                if appearance[i,j] > appearance_threshold:
                    appear_gate[i,j] = 1
                else: 
                    appear_gate[i,j] = 0
                    
                # 최종 cost 가중치 조합
                fin_gate[i,j] = appear_gate[i,j] * maha_dist_gate[i,j]
                if fin_gate[i,j] == 1:
                    cost_matrix[i, j] = alpha * appearance[i,j] + (1 - alpha) * maha_dist[i,j]
                else:
                    cost_matrix[i, j] = 1e5  

        # time_since_update 리스트 생성
        tracker_ages = [self.tracked_objects[t]['time_since_update'] for t in trackers]

        # 매칭된 트래커와 detection을 저장
        matches = []
        unmatched_trackers = list(range(len(trackers)))
        unmatched_detections = list(range(len(detections)))
        current_unmatched_dets = list(unmatched_detections)

        max_age = max(tracker_ages) if tracker_ages else 0
        for age in range(max_age + 1):
            # 현재 age 그룹에 해당하는 트랙 인덱스만 추출
            group_indices = [i for i, a in enumerate(tracker_ages) if a == age]
            if not group_indices:
                continue
            # 해당 트랙 인덱스를 원래 trackers 인덱스로 매핑
            group_tracker_indices = [trackers[i] for i in group_indices]

            # 서브 cost 행렬 추출: group_tracker_indices x unmatched_detections
            sub_cost = cost_matrix[np.ix_(group_indices, unmatched_detections)]
            if sub_cost.size == 0:
                continue

            # time since update을 기준으로 우선 배정
            row_ind, col_ind = linear_sum_assignment(sub_cost) # 헝가리안 알고리즘으로 최적 매칭

            for r, c in zip(row_ind, col_ind):
                trk_idx = group_tracker_indices[r]       # 전체 trackers 기준
                det_idx = current_unmatched_dets[c]        # 전체 detection 기준

                matches.append((trk_idx, det_idx))
                if trk_idx in unmatched_trackers:
                    unmatched_trackers.remove(trk_idx)
                if det_idx in unmatched_detections:
                    unmatched_detections.remove(det_idx)

        return matches, unmatched_trackers, unmatched_detections


    def update(self, frame,detections,iou_threshold=0.3, ema_alpha = 0.9):
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

        #for m in matches:
            #print("match types:", type(m[0]), type(m[1]))

        # 2. 매칭된 트래커 업데이트
        for t, d in matches:
            track = self.tracked_objects[t]
            detection = detections[d]
            if track['state'] == 'confirmed':
                track['time_since_update'] = 0
                track['bbox'] = track['kf'].update(detection['bbox'])
                track['hits'] +=1
                track['z_t_minus_2'] = track['z_t_minus_1']
                track['z_t_minus_1'] = detection['bbox']
                feat_track = np.array(track['feature'], dtype=np.float32)
                feat_det = np.array(detection['feature'], dtype=np.float32)
                track['feature'] = ema_alpha*feat_track + (1-ema_alpha)*feat_det  # DA !!
                feat_track = np.array(track['feature'], dtype=np.float32)
                track['feature'] /= np.linalg.norm(feat_track) + 1e-6 # L2 정규화

            if track['state'] == 'tentative':
                track['hits'] +=1
                track['time_since_update'] = 0
                track['bbox'] = track['kf'].update(detection['bbox'])
                track['z_t_minus_2'] = track['z_t_minus_1']
                feat_track = np.array(track['feature'], dtype=np.float32)
                feat_det = np.array(detection['feature'], dtype=np.float32)
                track['feature'] = ema_alpha*feat_track + (1-ema_alpha)*feat_det  # DA !!
                feat_track = np.array(track['feature'], dtype=np.float32)
                track['feature'] /= np.linalg.norm(feat_track) + 1e-6 # L2 정규화

                if track['hits'] >=3:
                    track['state'] = 'confirmed'

        # 3. 매칭 안 된 detection, trackers에 대해 두번째 매칭 (iou기반)
        cost_matrix_2 = np.zeros((len(unmatched_dets), len(unmatched_trks))) 
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
        for trk_idx,det_idx in matches_2:
            t = unmatched_trks[trk_idx]
            d = unmatched_dets[det_idx]
            trk = self.tracked_objects[t]
            det = detections[d]
            if trk['state'] =='tentative':
                trk['hits'] +=1
                trk['time_since_update'] = 0
                trk['bbox'] = trk['kf'].update(det['bbox'])
                trk['z_t_minus_2'] = trk['z_t_minus_1']
                trk['z_t_minus_1'] = det['bbox']
                feat_track = np.array(trk['feature'], dtype=np.float32)
                feat_det = np.array(det['feature'], dtype=np.float32)
                trk['feature'] = ema_alpha*feat_track + (1-ema_alpha)*feat_det  # DA !!
                feat_track = np.array(trk['feature'], dtype=np.float32)
                trk['feature'] /= np.linalg.norm(feat_track) + 1e-6 # L2 정규화

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
                'hits' : 1,
                'z_t_minus_1' : detections[d]['bbox'],
                'z_t_minus_2' : detections[d]['bbox'],
                'feature' : detections[d]['feature']
            }) 

        # 6. 트래커 삭제 
        max_age = 3
        for t in sorted(unmatched_trackers_2, reverse=True):
            self.tracked_objects[t]['time_since_update'] += 1
            #tentative 상태에서 삭제
            if self.tracked_objects[t]['state']=='tentative' and self.tracked_objects[t]['time_since_update'] >= 3:
                del self.tracked_objects[t] # 아예 관측되었던 object에서 삭제
            #confirm 상태에서 삭제
            elif self.tracked_objects[t]['state'] == 'confirmed' and self.tracked_objects[t]['time_since_update'] >= max_age:
                del self.tracked_objects[t]

        return self.tracked_objects

# ===================== 메인 =====================
if __name__ == "__main__":
    deepsort = Deepsort()
    cap = cv2.VideoCapture("dongwon_building-13.avi")

    frame_num = 0 

    total_frames = 0
    total_time = 0.0
    total_latency = 0.0
    extract_time = 0.0
    update_time = 0.0

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
        
        print('현재',frame_num,'번째 frame')
        t1 = time.time()
        detections = deepsort.extract_detections(frame)
        t2 = time.time()
        extract_time += (t2 - t1)       

        t3 = time.time()
        tracked = deepsort.update(frame, detections)
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
        cv2.imshow("Deepsort + YOLOv8 Tracking", frame)
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
