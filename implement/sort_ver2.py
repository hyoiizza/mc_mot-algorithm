import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import cv2
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

# YOLOv8 모델 로드
model = YOLO("yolov8n.pt")

# ===================== 칼만 상태 변환 =====================

# 바운딩 박스를 칼만 필터 상태로 변환
def bbox_to_kalman_state(bbox):
    u = bbox[0] + (bbox[2] - bbox[0]) / 2
    v = bbox[1] + (bbox[3] - bbox[1]) / 2
    s = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    r = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
    
    return np.array([u, v, s, r, 0, 0, 0])

# 칼만 필터 상태를 바운딩 박스로 변환
def kalman_state_to_bbox(state):
    u, v, s, r = state[0], state[1], state[2], state[3]
    w = np.sqrt(s * r)
    h = s / w
    x1 = u - w / 2
    y1 = v - h / 2
    x2 = u + w / 2
    y2 = v + h / 2

    return [x1, y1, x2, y2]

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

    # 칼만 필터 예측
    def predict(self):
        self.kf.predict()
        return kalman_state_to_bbox(self.kf.x)
    
    # 칼만 필터 업데이트
    def update(self, bbox):
        z = bbox_to_kalman_state(bbox)
        self.kf.update(z)
        return kalman_state_to_bbox(self.kf.x)

# ===================== SORT 클래스 =====================
class Sort:
    # SORT 알고리즘 초기화
    # 추적중인 객체 리스트와 추적 ID 초기화
    def __init__(self):
        self.tracked_objects = []
        self.track_id = 0

    # YOLO 모델 호출 후 탐지 수행.
    def detection(self, frame):
        return model(frame)

    # car와 truck 객체를 필터링 (2: car, 3: truck)
    # bbox와 confidence 정보를 포함한 detection 리스트 반환
    def extract_detections(self, frame):
        detections = []
        results = self.detection(frame)
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) in [2, 3]:  # car=2, truck=3
                    x1, y1, x2, y2 = box.xyxy[0]
                    detections.append({
                        'bbox': [x1.item(), y1.item(), x2.item(), y2.item()],
                        'confidence': box.conf[0].item()
                    })
        return detections
    '''
    기존 효정이 주석
    IoU 계산
        iou는 무조건 bbox형태의 인자만 받아야한다 !!
        box1 와 box2는 [x1, y1, x2, y2] 형태의 리스트
        box1은 칼만필터로 예측된 기존 tracker의 bbox
        box2는 새로운 detection의 bbox = detections[n]['bbox']
    '''
    # 두 박스의 IoU 계산(교집합 영역의 비율)
    def iou(self, box1, box2):
        xA = max(box1[0], box2[0]) # 교차 영역의 왼쪽 x 좌표
        yA = max(box1[1], box2[1]) # 교차 영역의 위쪽 y 좌표
        xB = min(box1[2], box2[2]) # 교차 영역의 오른쪽 x 좌표
        yB = min(box1[3], box2[3]) # 교차 영역의 아래쪽 y 좌표

        inter = max(0, xB - xA) * max(0, yB - yA) # 교집합 영역
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / float(area1 + area2 - inter + 1e-6) # 0으로 나누는 오류 방지

    '''
    기존 효정이 주석
    헝가리안 알고리즘을 이용한 매칭
        1. tracker와 detection의 모든 조합에 대해 IoU 계산
        2. IoU가 iou_threshold 이상인 경우 매칭 후보로 추가
        3. 헝가리안 알고리즘을 이용하여 최적 매칭
    '''

    # 매칭된 트래커와 새로운 detection을 비교하여 IoU 기반으로 매칭
    # 매칭된 트래커, 매칭되지 않은 트래커, 매칭되지 않은 detection 반환
    # 헝가리안 알고리즘으로 detections와 trackers를 매칭
    '''
    arguments
        iou_threshold = 0.3 (sort 논문값 이용)
        trackers = 칼만 필터로 예측된 기존 tracker의 상태 리스트 
    '''
    def match(self, detections, trackers, iou_threshold=0.3): #iou_threshold는 IoU 임계값

        if len(trackers) == 0: # 트래커가 없으면 매칭할 필요 없음
            return [], [], list(range(len(detections))) 

        cost_matrix = np.zeros((len(trackers), len(detections))) # 비용 행렬 초기화
        for i, trk in enumerate(trackers):
            for j, det in enumerate(detections):
                cost_matrix[i, j] = 1 - self.iou(trk['bbox'], det['bbox']) # IoU를 비용 행렬로 변환

        row_ind, col_ind = linear_sum_assignment(cost_matrix) # 헝가리안 알고리즘으로 최적 매칭

        # 매칭된 트래커와 detection을 저장
        matches = []
        unmatched_trackers = [list(range(len(trackers)))]
        unmatched_detections = list(range(len(detections))) 

        # 매칭된 트래커와 detection을 IoU 임계값에 따라 필터링
        for r, c in zip(row_ind, col_ind):
            if 1 - cost_matrix[r, c] > iou_threshold:
                matches.append((r, c))
                unmatched_trackers.remove(r)
                unmatched_detections.remove(c)

        # 매칭된 트래커와 detection을 반환
        return matches, unmatched_trackers, unmatched_detections

    '''
    핵심 로직
        1. 기존 트래커들 predict()
        2. 새 탐지와 트래커 매칭
        3. 매칭된 트래커는 update()
        4. 매칭 실패한 탐지는 새 트래커로 추가
        5. 매칭 실패한 기존 트래커는 삭제
    '''
    def update(self, detections):
        # 1. 기존 트래커 예측
        for trk in self.tracked_objects: # tracked_objects는 [{'id': id, 'kf': KalmanBox, 'bbox': [x1,y1,x2,y2]}, ...] 형태
            trk['bbox'] = trk['kf'].predict()

        matches, unmatched_trks, unmatched_dets = self.match(detections, self.tracked_objects)

        # 2. 매칭된 트래커 업데이트
        for t, d in matches:
            self.tracked_objects[t]['bbox'] = self.tracked_objects[t]['kf'].update(detections[d]['bbox'])

        # 3. 매칭 안 된 detection -> 새로운 트래커 생성
        for d in unmatched_dets:
            self.track_id += 1
            kf = KalmanBox(detections[d]['bbox'])
            self.tracked_objects.append({
                'id': self.track_id,
                'kf': kf,
                'bbox': detections[d]['bbox']
            })

        # 4. 매칭 안 된 tracker 삭제
        for t in sorted(unmatched_trks, reverse=True):
            del self.tracked_objects[t]

        return self.tracked_objects

# ===================== 메인 =====================
'''
효정이 코드 보니까 main 함수 코드에서 tracker랑 detections를 매칭하는 로직이 있는데
main 함수는 항상 딱 실행만 시키고, 추적 로직은 Sort 클래스 안에 구현되어 있어야해.

main함수는 볼줄 아니까 주석 안달을게 어제 효정이가 구현한 코드랑 이거랑 비교해봐.
실행도 시켜보고
근데, 그거 안넣었어 82-85줄에 있는 코드 'class_id' 빼먹었으니까
객체감지에 추가할거 추가해야 할 것 같아.

어제 고민했던 그 객체를 검출하고 매칭한다음 추출하는 부분을 update() 함수로 구현했어. (사실 gpt가 작성한 코드인데...)
그거를 근데 확실히 알고 넘어가야 할 것같아
이게 헝가리안 알고리즘이랑 칼만필터 각각 보면 이해하는데 이걸 SORT알고리즘으로 묶는거는 확실히
코드를 봐야 이해가 가능할거같아 그냥 생각해서 이해는 절대 못해
근데 이거 한번 이해하면 진짜 유용한게
영상처리 기반 객체 검출에서의 완전 기본기니까
코드 이해하고 효정이가 안보고 구현하는걸 한번 더 해보는게 좋을 것 같아
여보 똑똑해서 금방 이해 할거같아 난 이해하는걸 포기 했지만 (사실 거의 다 이해한듯 수학적인거 빼고 역시 효정이의 설명 GOAT)
그래도 주석 열심히 달아놓았으니까 한줄한줄 봐야해.

그리고 이거 코드창 띄우고 여보 코드창 띄우고 뭐가 잘못되었는지 어디부분이 약한지 비교하면서 스스로 분석을 해서 적어놔
그래야 다른 알고리즘이든 뭐든 구현 할 때 그게 도움이 될거야.
어디서 어떻게 구현을 해야지 틀, 그 틀을 잡을 수 있어.
이게 스스로 코드 분석 하는법이라서 클린코딩에 도움 돼;

나는 늦게 일어날거같으니까... 오늘 하루도 힘내 금욜이니까!! 사랑해!
'''

if __name__ == "__main__":
    sort = Sort()
    cap = cv2.VideoCapture("dongwon_building-09.avi")

    while cap.isOpened():
        success, frame = cap.read()
        # 효정이 main 코드 반영
        if frame is None:
            print("No frame to process.")
            break       

        if not success:
            print("Failed to read frame from video.")
            break

        detections = sort.extract_detections(frame) 
        tracked = sort.update(detections)

        # 추적 결과 시각화
        for trk in tracked:
            x1, y1, x2, y2 = [int(v) for v in trk['bbox']]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {trk['id']}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("SORT + YOLOv8 Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
