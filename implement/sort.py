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




class Sort:
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
        trackers = []    
        for i, state in enumerate(trackers):
            trackers.append({
                'bbox': kalman_state_to_bbox(state), 
                'class_id': state[4],  
                'object_index': i}) 
        # tracker는 [{'bbox': [x1,y1,x2,y2]},{'class_id': class_id},{'object_index': i}]

        cost_matrix = np.zeros((len(trackers), len(detections)))

        for i in range(len(trackers)):
            for j in range(len(detections)):
                cost_matrix[i,j] = 1 - self.iou(trackers[i]['bbox'], detections[j]['bbox'])
                if cost_matrix[i,j] < 1 - iou_threshold:
                    cost_matrix[i,j] = 1


        row_ind, col_ind = linear_sum_assignment(cost_matrix) # row_ind = tracker index, col_ind = detection index
        matched_indices = []  # 매칭된 인덱스 리스트     
        for i in range(len(row_ind)):
            if cost_matrix[row_ind[i], col_ind[i]] < 0.3:
                matched_indices.append((row_ind[i], col_ind[i]))

        # matched_indices는 [(tracker_index, detection_index), ...] 형태로 매칭된 인덱스 리스트
        
        # 구현 : detection index를 이용해 그에 매칭된 tracker index 정보를 덮어쓴다.  
        #     : 그리고 안 매칭된 detections에 남은 객체를 trackers 리스트에 추가한다. 
        #     : 매칭 안된 trackers는 history 리스트에 넣고, 다음 프레임에도 검출이 안되면 (같은 클래스id와 고유번호가 일치하지 않으면) 리스트에서 제거한다. 

        unmatced_trackers = []
        unmatced_detections = []

        trackers = 매칭된 놈 + unmatced_detections 

        return trackers, unmatced_trackers #반환 형태는 trackers, unmatched_trackers = [{'bbox': [x1,y1,x2,y2], 'class_id': class_id, 'object_index': i}, ...] 

        



class kalman:
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
    
    def predict(self,):
        ''' 칼만 필터 예측 '''
        # 이전 상태를 알고잇어야하네...
        self.kf.predict(이전상태..? 뭐지 그냥 저장되어있는건가?)
        return self.kf.x

    def update(self, bbox):
        ''' 칼만 필터 업데이트 '''
        z = bbox_to_kalman_state(bbox)
        self.kf.x = self.kf.update(z)   
        return self.kf.x    

    


    

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

        if not success:
            print("Failed to read frame from video.")
            break 
        # 한 프레임마다 전체 알고리즘 반복
        frame_num = 0

            
            results = sort.detection(frame)  # YOLOv8 모델을 이용한 객체 탐지
            if results:
                detections = sort.extract_detections(results)
                if not detections:
                    print("No relevant detections found.")
                    continue
            if frame_num == 0:
                # 첫 프레임에서는 칼만 필터 초기화 및 tracker 생성


            else:
            # 이후 프레임에서는 기존 tracker와 새로운 detections를 매칭
            # 매칭된 trackers를 칼만 필터로 업데이트
                trackers = 
            
                trackers, unmatced_trackers = sort.match(detections, sort.tracked_objects,iou_threshold=0.3)
                # 매칭된 trackers를 업데이트
                for tracker in trackers:
                    kf = kalman()
                    kf.update(tracker['bbox'])
                    tracker['bbox'] = kalman_state_to_bbox(kf.x)      

            frame_num += 1    

                # 결과 처리 및 시각화 로직 추가
                annotated_frame = results[0].plot()
                cv2.imshow("YOLOv8 Inference", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break
    
    cap.release()
    cv2.destroyAllWindows()


