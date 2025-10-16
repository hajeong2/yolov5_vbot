# yolo_udp_v5_unique_tracking.py
# ESP32-CAM UDP 수신 + YOLOv5 균열 감지 + 중복 제거 + Node-RED 비동기 전송

import socket
import cv2
import numpy as np
import torch
import time
import os
import requests
import base64
from threading import Thread
from queue import Queue
from collections import deque

# ==================== 설정 ====================
# YOLOv5 설정
REPO = r"E:/2003g/yolov5_temp"
WEIGHT = r"E:/2003g/yolov5_temp/runs/train/crack_finetune_temp/weights/best.pt"
SAVE_DIR = r"E:/2003g/yolov5_temp/runs/detected_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# 네트워크 설정
UDP_IP = "0.0.0.0"
UDP_PORT = 5000
NODE_RED_URL = "https://artistic-clearly-rooster.ngrok-free.app/yolo"

# 전송 최적화 설정
SEND_MAX_W = 640           # 전송용 이미지 최대 가로 크기 (픽셀)
JPEG_QUALITY = 70          # JPEG 압축 품질 (0-100, 낮을수록 용량 작음)
MIN_SEND_INTERVAL = 1.0    # 최소 전송 간격 (초)
MAX_QUEUE_SIZE = 5         # 전송 큐 최대 크기

# =====(추가) 전송 이미지 선택 =====
SEND_ANNOTATED = True      # True → 박스/라벨 그려진 annotated_img를 Node-RED로 전송

# =====(추가) 화면 주석 스타일 =====
DRAW_BOX = True
DRAW_LABEL = True          # False면 "crack" 텍스트 없이 박스만
LABEL_TEXT = "crack"
BOX_COLOR = (0, 255, 0)    # BGR
BOX_THICKNESS = 2
LABEL_FONT_SCALE = 0.6
LABEL_THICKNESS = 2

# ===== 중복 제거 설정 =====
IOU_THRESHOLD = 0.3        # 같은 균열로 판단하는 IoU 임계값 (낮을수록 엄격)
MEMORY_DURATION = 10.0     # 균열 기억 시간 (초) - 한번 본 균열은 이 시간동안 무시
MAX_MEMORY_SIZE = 100      # 최대 기억할 균열 수

# ==================== 균열 추적 클래스 ====================
class CrackTracker:
    """균열 중복 제거 및 유니크 카운트"""
    def __init__(self, iou_threshold=0.5, memory_duration=3.0, max_memory=50):
        self.iou_threshold = iou_threshold
        self.memory_duration = memory_duration
        self.max_memory = max_memory
        self.known_cracks = deque(maxlen=max_memory)  # [(bbox, timestamp), ...]
        self.unique_count = 0
    
    def calculate_iou(self, box1, box2):
        """두 박스의 IoU(Intersection over Union) 계산"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # 교집합 영역
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        inter_width = max(0, inter_xmax - inter_xmin)
        inter_height = max(0, inter_ymax - inter_ymin)
        inter_area = inter_width * inter_height
        
        # 합집합 영역
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0
        
        return inter_area / union_area
    
    def is_duplicate(self, bbox, current_time):
        """이 균열이 최근에 본 균열과 겹치는지 확인"""
        # 오래된 기억 삭제
        while self.known_cracks and (current_time - self.known_cracks[0][1]) > self.memory_duration:
            self.known_cracks.popleft()
        
        # 기존 균열과 비교
        for known_bbox, known_time in self.known_cracks:
            iou = self.calculate_iou(bbox, known_bbox)
            if iou > self.iou_threshold:
                return True  # 중복!
        
        return False  # 새로운 균열
    
    def add_cracks(self, detections_df, current_time):
        """새로운 감지 결과에서 유니크 균열만 추가"""
        new_cracks = 0
        
        for _, row in detections_df.iterrows():
            bbox = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])
            
            if not self.is_duplicate(bbox, current_time):
                self.known_cracks.append((bbox, current_time))
                self.unique_count += 1
                new_cracks += 1
        
        return new_cracks
    
    def get_stats(self):
        """통계 반환"""
        return {
            "unique_total": self.unique_count,
            "in_memory": len(self.known_cracks)
        }

# ==================== 전송 큐 및 워커 ====================
send_queue = Queue(maxsize=MAX_QUEUE_SIZE)
last_send_time = 0

def sender_worker():
    """백그라운드에서 Node-RED로 POST 요청 처리"""
    while True:
        payload = send_queue.get()
        if payload is None:  # 종료 신호
            break
        
        try:
            response = requests.post(NODE_RED_URL, json=payload, timeout=5)
            img_size_kb = len(payload['image_b64']) // 1024
            print(f"✓ [SENT] Status:{response.status_code} | New Cracks:{payload['new_cracks']} | Total:{payload['unique_total']} | Size:{img_size_kb}KB")
        except requests.Timeout:
            print(f"✗ [TIMEOUT] Node-RED 응답 없음")
        except requests.RequestException as e:
            print(f"✗ [ERROR] 전송 실패: {e}")
        except Exception as e:
            print(f"✗ [UNEXPECTED] {e}")
        finally:
            send_queue.task_done()

# 전송 스레드 시작 (데몬으로 실행)
sender_thread = Thread(target=sender_worker, daemon=True)
sender_thread.start()

# ==================== 유틸리티 함수 ====================
def shrink_for_send(img, max_w=SEND_MAX_W, quality=JPEG_QUALITY):
    """이미지를 전송용으로 축소 및 압축 후 base64 인코딩"""
    h, w = img.shape[:2]
    
    # 가로 크기 제한
    if w > max_w:
        ratio = max_w / w
        new_size = (max_w, int(h * ratio))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    
    # JPEG 압축
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encoded = cv2.imencode(".jpg", img, encode_param)
    
    if not success:
        raise RuntimeError("JPEG 인코딩 실패")
    
    # base64 변환
    b64_str = base64.b64encode(encoded.tobytes()).decode("ascii")
    return b64_str

# ==================== YOLOv5 모델 로드 ====================
print("YOLOv5 모델 로딩 중...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load(REPO, 'custom', path=WEIGHT, source='local').to(device)
model.conf = 0.35   # 신뢰도 임계값
model.iou = 0.45    # NMS IoU 임계값
model.max_det = 100 # 최대 감지 수

print(f"✓ 모델 로드 완료")
print(f"  - Device: {device}")
print(f"  - Classes: {model.names if hasattr(model, 'names') else 'Unknown'}")

# 균열 추적기 초기화
tracker = CrackTracker(IOU_THRESHOLD, MEMORY_DURATION, MAX_MEMORY_SIZE)
print(f"✓ 균열 추적 시스템 활성화")
print(f"  - IoU 임계값: {IOU_THRESHOLD}")
print(f"  - 기억 시간: {MEMORY_DURATION}초\n")

# ==================== UDP 소켓 설정 ====================
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
print(f"✓ UDP 서버 시작: {UDP_IP}:{UDP_PORT}")
print("ESP32-CAM 연결 대기 중...\n")

# JPEG 마커
SOI = b'\xff\xd8'  # Start of Image
EOI = b'\xff\xd9'  # End of Image
buffer = bytearray()

# 통계
frame_count = 0

# ==================== 메인 루프 ====================
try:
    while True:
        # UDP 패킷 수신
        data, addr = sock.recvfrom(65535)
        buffer += data
        
        # JPEG 이미지 추출
        start_idx = buffer.find(SOI)
        if start_idx != -1:
            end_idx = buffer.find(EOI, start_idx + 2)
            if end_idx != -1:
                # 완전한 JPEG 프레임 추출
                jpeg_data = bytes(buffer[start_idx:end_idx + 2])
                del buffer[:end_idx + 2]
                
                # JPEG 디코딩
                img = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                
                frame_count += 1
                current_time = time.time()
                
                # ===== YOLOv5 추론 =====
                results = model(img)
                detections_df = results.pandas().xyxy[0]

                # ===== 정확도 숫자 없이 직접 그리기 =====
                annotated_img = img.copy()
                if len(detections_df) > 0 and (DRAW_BOX or DRAW_LABEL):
                    for _, row in detections_df.iterrows():
                        x1, y1 = int(row['xmin']), int(row['ymin'])
                        x2, y2 = int(row['xmax']), int(row['ymax'])
                        if DRAW_BOX:
                            cv2.rectangle(annotated_img, (x1, y1), (x2, y2),
                                          BOX_COLOR, BOX_THICKNESS)
                        if DRAW_LABEL:
                            y_text = y1 - 8 if (y1 - 8) > 10 else (y1 + 20)
                            cv2.putText(annotated_img, LABEL_TEXT, (x1, y_text),
                                        cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE,
                                        BOX_COLOR, LABEL_THICKNESS, cv2.LINE_AA)
                
                # ===== 유니크 균열 추적 =====
                frame_crack_count = len(detections_df)  # 이 프레임에서 감지된 박스 수
                new_cracks = 0
                
                if frame_crack_count > 0:
                    # 새로운 균열만 카운트
                    new_cracks = tracker.add_cracks(detections_df, current_time)
                
                # ===== 새 균열 발견 시에만 전송 =====
                if new_cracks > 0:  # 진짜 새 균열이 있을 때만!
                    stats = tracker.get_stats()
                    timestamp = int(current_time)
                    
                    # 원본 이미지 저장(로컬)
                    save_path = os.path.join(SAVE_DIR, f"crack_{timestamp}_new{new_cracks}.jpg")
                    cv2.imwrite(save_path, img)
                        
                    # 전송용 이미지 준비 (annotated/원본 선택)
                    try:
                        frame_to_send = annotated_img if SEND_ANNOTATED else img
                        image_b64 = shrink_for_send(frame_to_send)
                        
                        # payload 생성
                        payload = {
                            "ts": int(current_time * 1000),
                            "crack_count": stats["unique_total"],   # 누적 유니크
                            "new_cracks": new_cracks,              # 이번 프레임 신규
                            "unique_total": stats["unique_total"], # 상세 동일
                            "frame_detections": frame_crack_count, # 이 프레임 박스 수
                            "image_b64": image_b64
                        }
                        
                        # 큐에 추가 (새 균열이 있으면 무조건 전송!)
                        if not send_queue.full():
                            send_queue.put(payload)
                            print(f"🆕 [NEW CRACK!] Frame:{frame_count} | New:{new_cracks} | Total Unique:{stats['unique_total']} | Memory:{stats['in_memory']}")
                        else:
                            print(f"⚠ [SKIP] 전송 큐 가득 차서 새 균열 전송 실패")
                    
                    except Exception as e:
                        print(f"✗ [ENCODE ERROR] {e}")
                
                # ===== 화면 표시 =====
                if frame_crack_count > 0:
                    stats = tracker.get_stats()
                    status_text = f"Frame:{frame_count} | Detected:{frame_crack_count} | Unique Total:{stats['unique_total']}"
                else:
                    stats = tracker.get_stats()
                    status_text = f"Frame:{frame_count} | No Cracks | Unique Total:{stats['unique_total']}"
                
                cv2.putText(annotated_img, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imshow("YOLOv5 Unique Crack Detection", annotated_img)
                
                # ESC 키로 종료
                if cv2.waitKey(1) & 0xFF == 27:
                    print("\n종료 요청됨...")
                    break

except KeyboardInterrupt:
    print("\n\nKeyboard Interrupt 감지...")

finally:
    # 정리 작업
    print("\n시스템 종료 중...")
    print(f"총 처리 프레임: {frame_count}")
    final_stats = tracker.get_stats()
    print(f"총 유니크 균열: {final_stats['unique_total']}")
    
    # 전송 큐 비우기
    send_queue.put(None)  # 종료 신호
    sender_thread.join(timeout=5)
    
    sock.close()
    cv2.destroyAllWindows()
    print("✓ 종료 완료")
