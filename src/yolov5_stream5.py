# yolo_udp_v5_unique_tracking.py
# ESP32-CAM UDP 수신 + YOLOv5 균열 감지 + 중복 제거 + (박스가 그려진) 이미지 그대로 Node-RED 전송
# - Node-RED 쪽 UI 변경 불필요 (img base64만 그대로 사용)
# - 렉 완화: 초당 1회 전송 cap, 큐=1, 비차단 put, timeout=1s, 최신 프레임만 처리, no_grad, (CUDA면) FP16

import socket
import cv2
import numpy as np
import torch
import time
import os
import requests
import base64
from threading import Thread
from queue import Queue, Full
from collections import deque

# ==================== 설정 ====================
# YOLOv5 설정
REPO   = r"E:/2003g/yolov5_temp"
WEIGHT = r"E:/2003g/yolov5_temp/runs/train/crack_finetune_temp/weights/best.pt"
SAVE_DIR = r"E:/2003g/yolov5_temp/runs/detected_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# 네트워크 설정
UDP_IP = "0.0.0.0"
UDP_PORT = 5000
NODE_RED_URL = "https://artistic-clearly-rooster.ngrok-free.app/yolo"

# 전송 최적화 설정
MIN_SEND_INTERVAL = 1.0     # 초당 최대 1회 전송
MAX_QUEUE_SIZE    = 1       # 전송 큐(막힘 방지)
REQUEST_TIMEOUT   = 1.0     # Node-RED 응답 제한

# ===== 전송 이미지(주석 포함) 크기/품질 =====
SEND_ANNOTATED    = True    # Node-RED에 보낼 이미지는 주석 포함본
ANN_MAX_W         = 480     # 주석 이미지는 480px로 축소(대역폭↓). 필요 시 640로 올려도 됨
ANN_JPEG_QUALITY  = 60      # 주석 이미지 JPEG 품질 (용량/렉 밸런스)

# ===== 화면 주석 스타일(내 로컬 미리보기) =====
DRAW_BOX          = True
DRAW_LABEL        = False   # 정확도 숫자 제거 (원하면 True로)
LABEL_TEXT        = "crack"
BOX_COLOR         = (0, 255, 0)   # BGR
BOX_THICKNESS     = 2
LABEL_FONT_SCALE  = 0.6
LABEL_THICKNESS   = 2

# ===== 중복 제거(유니크 카운트) =====
IOU_THRESHOLD   = 0.3
MEMORY_DURATION = 10.0
MAX_MEMORY_SIZE = 100

# ===== UDP 수신 버퍼 세이프티 =====
RECVBUF_MAX = 2 * 1024 * 1024  # 2MB 넘으면 버퍼 클리어(낡은 프레임 과감히 버리기)

# ==================== 균열 추적 클래스 ====================
class CrackTracker:
    def __init__(self, iou_threshold=0.5, memory_duration=3.0, max_memory=50):
        self.iou_threshold = iou_threshold
        self.memory_duration = memory_duration
        self.max_memory = max_memory
        self.known_cracks = deque(maxlen=max_memory)  # [(bbox, timestamp)]
        self.unique_count = 0
    
    def _iou(self, b1, b2):
        x1a, y1a, x2a, y2a = b1
        x1b, y1b, x2b, y2b = b2
        ix1, iy1 = max(x1a, x1b), max(y1a, y1b)
        ix2, iy2 = min(x2a, x2b), min(y2a, y2b)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        a1 = (x2a - x1a) * (y2a - y1a)
        a2 = (x2b - x1b) * (y2b - y1b)
        union = a1 + a2 - inter
        return 0 if union == 0 else inter / union
    
    def is_dup(self, bbox, t):
        # 오래된 기억 삭제
        while self.known_cracks and (t - self.known_cracks[0][1]) > self.memory_duration:
            self.known_cracks.popleft()
        for kb, _ in self.known_cracks:
            if self._iou(bbox, kb) > self.iou_threshold:
                return True
        return False
    
    def add(self, df, t):
        new_cracks = 0
        for _, r in df.iterrows():
            b = (r['xmin'], r['ymin'], r['xmax'], r['ymax'])
            if not self.is_dup(b, t):
                self.known_cracks.append((b, t))
                self.unique_count += 1
                new_cracks += 1
        return new_cracks
    
    def stats(self):
        return {"unique_total": self.unique_count, "in_memory": len(self.known_cracks)}

# ==================== 전송 큐/워커 ====================
send_queue = Queue(maxsize=MAX_QUEUE_SIZE)
last_send_time = 0.0

def sender_worker():
    while True:
        payload = send_queue.get()
        if payload is None:
            break
        try:
            r = requests.post(NODE_RED_URL, json=payload, timeout=REQUEST_TIMEOUT)
            kb = len(payload.get('image_b64', "")) // 1024
            print(f"✓ [SENT] HTTP:{r.status_code} | New:{payload['new_cracks']} | Total:{payload['unique_total']} | {kb}KB")
        except requests.Timeout:
            print("✗ [TIMEOUT] Node-RED 응답 없음")
        except requests.RequestException as e:
            print(f"✗ [ERROR] 전송 실패: {e}")
        finally:
            send_queue.task_done()

Thread(target=sender_worker, daemon=True).start()

# ==================== 유틸 함수 ====================
def encode_jpeg_b64(img, max_w, quality):
    h, w = img.shape[:2]
    if w > max_w:
        r = max_w / w
        img = cv2.resize(img, (max_w, int(h * r)), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG 인코딩 실패")
    return base64.b64encode(enc.tobytes()).decode("ascii")

# ==================== YOLOv5 모델 ====================
print("YOLOv5 모델 로딩 중...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load(REPO, 'custom', path=WEIGHT, source='local').to(device)
if device == 'cuda':
    model.half()  # FP16
torch.backends.cudnn.benchmark = True
model.conf = 0.35
model.iou  = 0.45
model.max_det = 100
print("✓ 모델 로드 완료")
print(f"  - Device: {device}")
print(f"  - Classes: {getattr(model, 'names', 'Unknown')}")

tracker = CrackTracker(IOU_THRESHOLD, MEMORY_DURATION, MAX_MEMORY_SIZE)
print("✓ 균열 추적 활성화")
print(f"  - IoU: {IOU_THRESHOLD} | 메모리: {MEMORY_DURATION}s\n")

# ==================== UDP 소켓 ====================
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(0.2)  # 블로킹 방지
print(f"✓ UDP 서버 시작: {UDP_IP}:{UDP_PORT}")
print("ESP32-CAM 연결 대기...\n")

SOI, EOI = b'\xff\xd8', b'\xff\xd9'
buffer = bytearray()
frame_count = 0

# ==================== 메인 루프 ====================
try:
    while True:
        # ---- 수신 (비블로킹) ----
        try:
            data, addr = sock.recvfrom(65535)
            buffer += data
        except socket.timeout:
            pass

        # 버퍼 과도 시 정리(낡은 데이터 드랍)
        if len(buffer) > RECVBUF_MAX:
            print("⚠ buffer overflow → clear")
            buffer.clear()
            continue

        # ---- 최신 완전 프레임만 추출 ----
        end_idx = buffer.rfind(EOI)
        if end_idx == -1:
            continue
        start_idx = buffer.rfind(SOI, 0, end_idx)
        if start_idx == -1:
            # SOI가 없으면 앞부분 폐기
            del buffer[:end_idx+2]
            continue

        jpeg = bytes(buffer[start_idx:end_idx+2])
        del buffer[:end_idx+2]

        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue

        frame_count += 1
        now = time.time()

        # ===== YOLO 추론 =====
        with torch.no_grad():
            results = model(img)
        df = results.pandas().xyxy[0]

        # ===== 화면용 주석(텍스트 없이 박스만) =====
        annotated = img.copy()
        if len(df) > 0 and (DRAW_BOX or DRAW_LABEL):
            for _, r in df.iterrows():
                x1, y1 = int(r['xmin']), int(r['ymin'])
                x2, y2 = int(r['xmax']), int(r['ymax'])
                if DRAW_BOX:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
                if DRAW_LABEL:
                    y_text = y1 - 8 if (y1 - 8) > 10 else (y1 + 20)
                    cv2.putText(annotated, LABEL_TEXT, (x1, y_text),
                                cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE,
                                BOX_COLOR, LABEL_THICKNESS, cv2.LINE_AA)

        # ===== 유니크 추적 & 통계 =====
        frame_crack_count = len(df)
        new_cracks = tracker.add(df, now) if frame_crack_count > 0 else 0
        stats = tracker.stats()

        # ===== 전송 (새 균열 + 간격 제한) =====
        if SEND_ANNOTATED and new_cracks > 0 and (now - last_send_time) >= MIN_SEND_INTERVAL:
            # 로컬 저장
            cv2.imwrite(os.path.join(SAVE_DIR, f"crack_{int(now)}_new{new_cracks}.jpg"), annotated)
            try:
                image_b64 = encode_jpeg_b64(annotated, ANN_MAX_W, ANN_JPEG_QUALITY)
                payload = {
                    "ts": int(now * 1000),
                    "image_b64": image_b64,
                    "new_cracks": new_cracks,
                    "unique_total": stats["unique_total"],
                    "frame_detections": frame_crack_count
                }
                try:
                    send_queue.put_nowait(payload)   # 큐 꽉 차면 드롭(대기 금지)
                    last_send_time = now
                    print(f"🆕 [NEW CRACK] F:{frame_count} | New:{new_cracks} | Total:{stats['unique_total']}")
                except Full:
                    print("⚠ [SKIP] 전송 큐 가득(최신만 유지)")
            except Exception as e:
                print(f"✗ [ENCODE ERROR] {e}")

        # ===== 화면 표시 =====
        status = f"Frame:{frame_count} | {'Detected:'+str(frame_crack_count) if frame_crack_count else 'No Cracks'} | Unique:{stats['unique_total']}"
        cv2.putText(annotated, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imshow("YOLOv5 Unique Crack Detection", annotated)
        if cv2.waitKey(1) & 0xFF == 27:
            print("\n종료 요청됨...")
            break

except KeyboardInterrupt:
    print("\n\nKeyboard Interrupt 감지...")

finally:
    print("\n시스템 종료 중...")
    print(f"총 처리 프레임: {frame_count}")
    print(f"총 유니크 균열: {tracker.stats()['unique_total']}")
    send_queue.put(None)
    time.sleep(0.1)
    sock.close()
    cv2.destroyAllWindows()
    print("✓ 종료 완료")
