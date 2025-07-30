import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from encoder_model  import DeepSortEncoder


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
    s = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    r = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
    
    return np.array([u, v, s, r, 0, 0, 0, 0])

def kalman_state_to_bbox(state):
    # 칼만 필터 상태를 바운딩 박스로 변환
    u, v, s, r = state[0], state[1], state[2], state[3]
    
    w = np.sqrt(s * r)
    h = s / w

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
        self.kf = KalmanFilter(dim_x=7, dim_z=4) # 칼만 필터 초기화
        self.kf.x = bbox_to_kalman_state(bbox) # 칼만 필터 상태 초기화
        self.kf.P *= 1000. # 초기 오차 공분산 행렬
        self.kf.F = np.eye(7) # 상태 전이 행렬
        for i in range(4):
            self.kf.F[i-4, i] = 1 # 칼만 필터 상태 전이 행렬
        self.kf.H = np.zeros((4, 7)) # 관측 모델 행렬
        self.kf.H[0, 0] = 1 # u
        self.kf.H[1, 1] = 1 # v
        self.kf.H[2, 2] = 1 # gamma
        self.kf.H[3, 3] = 1 # h
        # 공분산 설정
        self.kf.R *= 10
        self.kf.Q *= 0.01 

    def predict(self):


        try:
            # 칼만 필터 예측
            s_before = self.kf.x[2]
            ds_before = self.kf.x[6]
            self.kf.predict()

        except ValueError as e:
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
            if s_after <= 0:
                raise ValueError(f"[PREDICT ERROR] in || kalman predict (s_predicted={s_after}) | (s_before={s_before}, ds={ds_before})")
            if r <= 0:
                raise ValueError(f"[PREDICT ERROR] r is non-positive: r={r}")

        return kalman_state_to_bbox(self.kf.x)
    
    def update(self, bbox):
        # 칼만 필터 업데이트
        z = bbox_to_kalman_state(bbox)
        z = np.atleast_2d(z[:4]).T

        try:
            # 칼만 필터 예측
            s_before = self.kf.x[2]
            ds_before = self.kf.x[6]
            self.kf.update(z)

        except ValueError as e:
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
            if s_after <= 0:
                raise ValueError(f"[PREDICT ERROR] in || kalman update (s_predicted={s_after}) | (s_before={s_before}, ds={ds_before})")
            if r <= 0:
                raise ValueError(f"[PREDICT ERROR] r is non-positive: r={r}")
        
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

# ===================== Deep OCSORT 클래스 ====================
class DeepOCsort:
    def __init__(self):
        # SORT 알고리즘 초기화
        # 추적중인 객체 리스트와 추적 ID 초기화
        self.tracked_objects = []
        self.track_id = 0
        self.z_t_minus_2 = None
        self.z_t_minus_1 = None
        self.prev_frame = None
        self.camera_motion = np.array([0, 0])  # 초기 이동량


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
    
    def compute_ema_alpha(self, s_det, sigma=0.5, alpha_f=0.95):
        if s_det <= sigma:
            return 1.0
        alpha_t = alpha_f + (1-alpha_f)*(1-(s_det-sigma)/(1-sigma))
        return min(alpha_t, 1.0)

    def compute_aw(self, appearance, epsilon=0.5):
        M,N = appearance.shape
        z_track = np.zeros(M)
        z_det = np.zeros(N)

        for i in range(M):
            row = appearance[i, :]
            top1 = np.max(row)
            top2 = np.max(row[row < top1] if np.any(row < top1) else top1)
            z_track[i] = min(top1-top2,epsilon)
        for j in range(N):
            col = appearance[:,j]
            top1 = np.max(col)
            top2 = np.max(col[col<top1] if np.any(col<top1) else top1)
            z_det[j] = min(top1 - top2, epsilon)
        
        w = (z_track[:, None] + z_det[None, :]) / 2.0
        return w

    def compute_camera_motion(self, prev_frame, curr_frame):
        # Optical flow로 카메라 이동량 추정
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        # 전체 평균 이동량 (dx, dy)
        dx = np.mean(flow[..., 0])
        dy = np.mean(flow[..., 1])
        return np.array([dx, dy])

    def apply_camera_motion(self, bbox, motion):
        x1, y1, x2, y2 = bbox
        return [x1 - motion[0], y1 - motion[1], x2 - motion[0], y2 - motion[1]]


    def match(self, frame, detections, trackers, appearance_weight=0.5,iou_threshold=0.3 ,appearance_threshold=0.5):
        if len(trackers) == 0: # 트래커가 없으면 매칭할 필요 없음
            return [], [], list(range(len(detections))) 

        trackers = list(range(len(self.tracked_objects)))

        cost_matrix = np.zeros((len(trackers), len(detections))) 
        cost_iou = np.zeros((len(trackers), len(detections))) 
        appearance = np.zeros((len(trackers), len(detections)))
        appear_gate = np.zeros((len(trackers), len(detections)))
        iou_gate = np.zeros((len(trackers), len(detections)))
        fin_gate = np.zeros((len(trackers), len(detections)))

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


        for i, trk in enumerate(trackers):
            trk = self.tracked_objects[trk]
            for j, det in enumerate(detections):
                cost_iou[i, j] = 1 - self.iou(trk['bbox'], det['bbox']) 
                appearance[i,j] = 1 - cosine_similarity(detection_features[j], tracker_features[i])
                if cost_iou[i,j] >= iou_threshold:
                    iou_gate[i,j] = 1
                else:
                    iou_gate[i,j] = 0
                if appearance[i,j] > appearance_threshold:
                    appear_gate[i,j] = 1
                else: 
                    appear_gate[i,j] = 0 
                fin_gate[i,j] = appear_gate[i,j]*iou_gate[i,j]
                w = self.compute_aw(appearance) # AW !!
                if fin_gate[i,j] == 1:
                    cost_matrix[i, j] = cost_iou[i,j] + (appearance_weight + w[i,j])*appearance[i,j] # AW !!
                else:
                    cost_matrix[i, j] = 1e5 

        row_ind, col_ind = linear_sum_assignment(cost_matrix) # 헝가리안 알고리즘으로 최적 매칭

        # 매칭된 트래커와 detection을 저장
        matches = []
        unmatched_trackers = list(range(len(trackers)))
        unmatched_detections = list(range(len(detections))) 

        # 매칭된 트래커와 detection을 IoU 임계값에 따라 필터링
        for r, c in zip(row_ind, col_ind):
            matches.append((r, c))
            unmatched_trackers.remove(r)
            unmatched_detections.remove(c)

        return matches, unmatched_trackers, unmatched_detections


    def update(self, frame ,detections ,frame_num, ema_alpha = 0.9):
        '''
            핵심 로직
            1. 기존 트래커들 predict()
            2. 매칭된 트래커 업데이트
            3. re-id로 매칭된 tracker의 가상 궤적 보정 (OCR)
            4. 두번째 매칭에 의해 새로 발견한 객체 업데이트(new track)
            5. 트래커 삭제
        '''
        if self.prev_frame is not None:
            self.camera_motion = self.compute_camera_motion(self.prev_frame, frame)

        # 1. 기존 트래커 예측
        for trk in self.tracked_objects: # tracked_objects는 [{'id': id, 'kf': KalmanBox, 'bbox': [x1,y1,x2,y2]}, ...] 형태
            try:
                trk['bbox'] = trk['kf'].predict()
                trk['bbox'] = self.apply_camera_motion(trk['bbox'], self.camera_motion) # CMC(OCM)
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
        
        matches, unmatched_trks, unmatched_dets = self.match(frame, detections, self.tracked_objects)

        #for m in matches:
            #print("match types:", type(m[0]), type(m[1]))

        re_id = []
        ema_alpha = ema_alpha
        # 2. 매칭된 트래커 업데이트
        for t, d in matches:
            track = self.tracked_objects[t]
            detection = detections[d]
            s_det = detection['confidence']
            ema_alpha = self.compute_ema_alpha(s_det, sigma=0.5, alpha_f=0.95) # DA 구현
            if track['time_since_update'] >=1: # re-id의 첫 번째 조건
                re_id.append((t,d))
            else:
                track['time_since_update'] = 0
                track['bbox'] = track['kf'].update(detection['bbox'])
                track['z_t_minus_2'] = track['z_t_minus_1']
                track['z_t_minus_1'] = detection['bbox']
                feat_track = np.array(track['feature'], dtype=np.float32)
                feat_det = np.array(detection['feature'], dtype=np.float32)
                track['feature'] = ema_alpha*feat_track + (1-ema_alpha)*feat_det  # DA !!
                feat_track = np.array(track['feature'], dtype=np.float32)
                track['feature'] /= np.linalg.norm(feat_track) + 1e-6 # L2 정규화

         # 3. re-id로 매칭된 tracker의 가상 궤적 보정 (OCR)
        for t,d in re_id:
            track = self.tracked_objects[t]
            detection = detections[d]
            z_start = track['z_t_minus_1'] # re-id 되기 전의 마지막 tracking 값
            z_end = detection['bbox'] # re-id 된 tracking 값
            num_missing_frame = frame_num - track['start_missing_frame_num'] + 1 # 객체가 사라진 frame수 
            feat_track = np.array(track['feature'], dtype=np.float32)
            feat_det = np.array(detection['feature'], dtype=np.float32)
            virtual_obs_list = []
            for i in range(1,num_missing_frame+1):
                alpha = i / (num_missing_frame+1)
                z_virtual = (1 - alpha) * np.array(z_start) + alpha * np.array(z_end)
                virtual_obs_list.append(z_virtual)
            for z_virtual in virtual_obs_list:
                track['kf'].predict()
                z_virtual_cm = self.apply_camera_motion(z_virtual, self.camera_motion)  # <-- CMC (OCR)
                track['bbox'] = track['kf'].update(z_virtual_cm)

            # 마지막 진짜 detection으로 업데이트
            z_end_cm = self.apply_camera_motion(z_end, self.camera_motion)
            track['kf'].predict()
            track['bbox'] = track['kf'].update(z_end_cm) # CMC (OOS)
            track['time_since_update'] = 0
            track['z_t_minus_2'] = track['z_t_minus_1']
            track['z_t_minus_1'] = z_end
            track['start_missing_frame_num'] = 're-identified!'
            track['feature'] = ema_alpha*feat_track+ (1-ema_alpha)*feat_det
            track['feature'] /= np.linalg.norm(feat_track) + 1e-6 # L2 정규화

        # 4. 두번째 매칭에 의해 새로 발견한 객체 업데이트(new track)
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
                'start_missing_frame_num': 'not-yet',
                'z_t_minus_1' : detections[d]['bbox'],
                'z_t_minus_2' : detections[d]['bbox'],
                'feature' : np.zeros(128)
            }) 
        # 5. 트래커 삭제 
        max_age = 3
        for t in sorted(unmatched_trks, reverse=True):
            self.tracked_objects[t]['time_since_update'] += 1
            self.tracked_objects[t]['start_missing_frame_num'] = frame_num

            if self.tracked_objects[t]['time_since_update'] >= max_age:
                del self.tracked_objects[t] # 아예 관측되었던 object에서 삭제

        self.prev_frame = frame.copy()

        return self.tracked_objects

# ===================== 메인 =====================
if __name__ == "__main__":
    deepocsort = DeepOCsort()
    cap = cv2.VideoCapture("dongwon_building-09.avi")

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
        
        print('현재',frame_num,'번째 frame')

        total_frames += 1
        frame_start = time.time()

        t1 = time.time()
        detections = deepocsort.extract_detections(frame)
        t2 = time.time()
        extract_time += (t2 - t1)

        t3 = time.time()
        tracked = deepocsort.update(frame, detections, frame_num)
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
