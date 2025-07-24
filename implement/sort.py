'''
=======================================
[Sort model]
1. detection : yolo v8 모델 이용
2. tracking : SORT 알고리즘 이용
========================================

========================================
class : sort
========================================
함수 설명
1. __init__ : SORT 알고리즘 초기화
2. detection : YOLOv8 모델을 이용한 객체 탐지 0
3. detection 결과에서 car, truck 만 추출 0 
4. IoU 계산 0 
5. 헝가리안 알고리즘을 이용한 매칭 (tracker랑 객체 탐지 결과 매칭) --> 3가지 상태의 리스트 반환 0
========================================
class : KalmanFilter
=========================================
1. __init__ : 칼만 필터 초기화
2. predict : 칼만 필터 예측
3. update : 칼만 필터 업데이트
4. track_init : 새로운 객체 추적 초기화
5. track : 객체 추적
========================================
class : MCMOT (Multi-Class Multi-Object Tracking)
sort 클래스에서 수행한 Mot결과를 Multi Camera로 연결

========================================
'''

import numpy as np
import cv2
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


model = YOLO("yolov8n.pt")

def bbox_to_kalman_state(bbox):
    ''' 바운딩 박스를 칼만 필터 상태로 변환 '''
    u = bbox[0] + (bbox[2] - bbox[0]) / 2
    v = bbox[1] + (bbox[3] - bbox[1]) / 2
    s = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    r = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
    return np.array([u, v, s, r, 0, 0, 0])  

def kalman_state_to_bbox(state):
    ''' 칼만 필터 상태를 바운딩 박스로 변환 '''

    #state = np.array([u, v, s, r, u_dot, v_dot, s_dot])

    u = state[0]  # u는 bbox의 중심 x 좌표
    v = state[1]  # v는 bbox의 중심 y 좌표
    s = state[2] # s는 bbox의 면적
    r = state[3] # r은 bbox의 가로 세로 비율
        
    w = np.sqrt(s * r)
    h = s / w  

    x1 = u - w / 2
    y1 = v - h / 2
    x2 = u + w / 2
    y2 = v + h / 2

    return [x1, y1, x2, y2]




class Sort():
    def __init__(self, ):
        ''' SORT 알고리즘 초기화 '''
        self.tracked_objects = []  # 추적 중인 객체 리스트  
        pass

    def detection(self, frame):
        ''' YOLOv8 모델을 이용한 객체 탐지 '''
        results = model(frame)
        return results
    
    def extract_detections(self):
        ''' detection 결과에서 car, truck 만 추출 '''
        detections = []
        results = self.detection(frame)  # YOLOv8 모델을 이용한 객체 탐지
        for result in results:
            for box in result.boxes:
                if box.cls in [2, 3]:  # 2: car, 3: truck
                    x1, y1, x2, y2 = box.xyxy[0] # 좌표 추출
                    confidence = box.conf[0] # 신뢰도 추출
                    detections.append({
                        'bbox': [x1.item(), y1.item(), x2.item(), y2.item()],
                        'confidence': confidence.item(),
                        'class_id': int(box.cls[0]) 
                    })

        return detections # 반환된 detections는 리스트 형태로, 각 객체의 bbox, confidence, class_id를 포함한다. 


    def iou(self, box1, box2):
        ''' 
        IoU 계산
        iou는 무조건 bbox형태의 인자만 받아야한다 !!
        box1 와 box2는 [x1, y1, x2, y2] 형태의 리스트
        box1은 칼만필터로 예측된 기존 tracker의 bbox
        box2는 새로운 detection의 bbox = detections[n]['bbox']
        '''

        x_a = max(box1[0], box2[0])
        y_a = max(box1[1], box2[1])
        x_b = min(box1[2], box2[2])
        y_b = min(box1[3], box2[3])

        intersection_area = max(0, x_b - x_a) * max(0, y_b - y_a)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])   

        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        
        return iou


    def match(self, detections, trackers, iou_threshold=0.3):
        ''' 
        헝가리안 알고리즘을 이용한 매칭
        1. tracker와 detection의 모든 조합에 대해 IoU 계산
        2. IoU가 iou_threshold 이상인 경우 매칭 후보로 추가
        3. 헝가리안 알고리즘을 이용하여 최적 매칭
        
        arguments
        iou_threshold = 0.3 (sort 논문값 이용)
        trackers = 칼만 필터로 예측된 기존 tracker의 상태 리스트 
        '''
        tracker = []    
        for i, state in enumerate(trackers):
            tracker.append({
                'bbox': kalman_state_to_bbox(state), 
                'class_id': state[4],  
                'object_index': i}) 
        # tracker는 [{'bbox': [x1,y1,x2,y2]},{'class_id': class_id},{'object_index': i}]

        print("||","tracker 수:", len(tracker) ,"|","검출된 객체 수:",len(detections)) # 디버그용
        
        for i in range(len(tracker)):
            for j in range(len(detections)):
                iou = self.iou(tracker[i]['bbox'], detections[j]['bbox'])
                if iou < iou_threshold:
                    continue
                # IoU가 임계값 이상인 경우 매칭 후보로 추가
                tracker[i]['iou'] = iou
                tracker[i]['detection_index'] = j


        # 헝가리안 알고리즘을 이용하여 최적 매칭
        cost_matrix = np.zeros((len(tracker), len(detections)))
        for i in range(len(tracker)):
            for j in range(len(detections)):
                cost_matrix[i, j] = 1 - tracker[i]['iou'] if 'iou' in tracker[i] else 1
        row_ind, col_ind = linear_sum_assignment(cost_matrix) # row_ind = tracker index, col_ind = detection index
        
        





            
    
    
'''
sort 클래스 최종 반환 형태
matched_indices = [(tracker_index, detection_index), ...] 
unmacthced_trackers = []
unmatched_detections = []
'''



class kalman():
    def __init__(self):
        ''' 
        칼만 필터 초기화 
        상태벡터 = [u, v, s, r, u_dot, v_dot, s_dot]
        '''
        self.kf = KalmanFilter(dim_x=7, dim_z=4) #칼만필터 생성 ㅣ 상태 벡터(x) 7, 관측 벡터(z) 4
        self.kf.x = np.zeros(7)  # 상태 벡터 초기화
        self.kf.P *= 1000.  # 초기 공분산 행렬
        self.kf.F = np.eye(7)  # 상태 전이 행렬
        self.kf.H = np.zeros((4, 7))  # 관측 행렬
        self.kf.R = np.eye(4) * 10  # 관측 잡음 공분산 행렬
        self.kf.Q = np.eye(7) * 0.01  # 프로세스 잡음 공분산 행렬 
        self.kf.H[0, 0] = 1  # u
        self.kf.H[1, 1] = 1  # v    
        self.kf.H[2, 2] = 1  # s
        self.kf.H[3, 3] = 1  # r 

    def track_init(self, bbox, class_id):
        ''' 새로운 객체 추적 초기화 '''
        self.kf.x[0] = bbox[0] + (bbox[2] - bbox[0]) / 2    
        self.kf.x[1] = bbox[1] + (bbox[3] - bbox[1]) / 2
        self.kf.x[2] = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])  # s
        self.kf.x[3] = (bbox[2] - bbox[0])
        self.kf.x[4] = class_id  # class
        self.kf.x[5] = 0  # u_dot
        self.kf.x[6] = 0  # v_dot
    
    def predict(self):
        ''' 칼만 필터 예측 '''
        self.kf.predict()
        return self.kf.x

    def update(self, bbox):
        ''' 칼만 필터 업데이트 '''
        z = bbox_to_kalman_state(bbox)
        self.kf.x = self.kf.update(z)   
        return self.kf.x    


class tracked_list(Sort,kalman):
    def __init__(self, ):
        self.tracked_objects = []  # 추적 중인 객체 리스트


if  __name__ == "__main__":
    sort = Sort()
    
    # 비디오 스트림에서 프레임을 읽어오는 예시
    video_path = 'dongwon_building-09.avi'
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        success, frame = cap.read()
        if frame is None:
            print("No frame to process.")
            break       


        results = sort.detection(frame)  # YOLOv8 모델을 이용한 객체 탐지
        if results:
            detections = sort.extract_detections(results)
            if not detections:
                print("No relevant detections found.")
                continue
        



            # 결과 처리 및 시각화 로직 추가
            annotated_frame = results[0].plot()
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()

    ----------------------
지금 너무 헷갈리는게 
detections = [{bbox: [],confidence:n, class_id :n, 객체 고유번호: n}, {...}, ]
 
track_object = [ {state : [7개], class_id: n, 객체 고유번호 : n}]
trackers = [track_object(), track_object(), ...]