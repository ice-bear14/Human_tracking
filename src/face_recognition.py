# Import Library
import cv2
import numpy as np
import requests
import threading
import time
import torch
import psutil
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Info Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Device digunakan: {device}")

# Model yang Digunakan
model = YOLO(r'C:\Users\panji\OneDrive\Documents\Propo + Skripsi\Kode Alat Fix\best.pt')
model.to(device)
model.fuse()
model.overrides['agnostic_nms'] = True
model.half()

tracker = DeepSort(max_age=30)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(r'C:\Users\panji\OneDrive\Documents\Propo + Skripsi\Kode Alat Fix\model_wajah.yml')
names = ['None', 'Panji', 'Moto-Moto', 'Gappar', 'Alpin']

# URL Raspi
RASPI_URL = "http://192.168.18.203:5000/pwm"

def send_pwm_async(pwm_x, pwm_y):
    """Kirim data PWM non-blocking"""
    def worker():
        try:
            requests.post(RASPI_URL, json={"pwm_x": int(pwm_x), "pwm_y": int(pwm_y)}, timeout=0.05)
            print(f"[SEND] pwm_x={pwm_x:.1f}, pwm_y={pwm_y:.1f}")
        except Exception as e:
            print(f"[SEND_FAIL] pwm_x={pwm_x:.1f}, pwm_y={pwm_y:.1f}, err={type(e).__name__}")
    threading.Thread(target=worker, daemon=True).start()

# Deadzone
DEADZONE = 60
SMOOTH_FACTOR = 0.05        
FACE_SMOOTH_FACTOR = 0.2    
AUTO_CENTER_DELAY = 3.0     

servo_x, servo_y = 90.0, 90.0
last_face_time = time.time()
last_face_cx, last_face_cy = None, None
status_text = "WAITING"

def calculate_pwm_target(cx, cy, frame_w, frame_h, deadzone=DEADZONE):
    """Hitung PWM target berdasarkan posisi wajah (arah vertikal dibalik)"""
    center_x, center_y = frame_w // 2, frame_h // 2
    offset_x, offset_y = cx - center_x, cy - center_y

    # Antisipasi pergerakan piksel di deadzone
    if abs(offset_x) < deadzone and abs(offset_y) < deadzone:
        return None, None

    # Arah Servo
    new_x = np.clip(90 + offset_x * 0.15, 0, 180)
    new_y = np.clip(90 + offset_y * 0.10, 45, 135)
    return new_x, new_y


# Recognisi atau Pengenalan Wajah
face_to_recognize = None
recognized_name = "Unknown"
recognized_conf = 0.0
recognition_lock = threading.Lock()

def recognize_face_worker():
    global face_to_recognize, recognized_name, recognized_conf
    while True:
        if face_to_recognize is not None and face_to_recognize.size > 0:
            try:
                gray = cv2.cvtColor(face_to_recognize, cv2.COLOR_BGR2GRAY)
                label, conf = recognizer.predict(gray)
                with recognition_lock:
                    if conf < 100:
                        recognized_name = names[label]
                        recognized_conf = conf
                    else:
                        recognized_name = "Unknown"
                        recognized_conf = conf
            except Exception:
                recognized_name = "Error"
                recognized_conf = 0
            face_to_recognize = None
        time.sleep(0.02)

threading.Thread(target=recognize_face_worker, daemon=True).start()

# Inisialisasi Kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Kamera tidak bisa dibuka.")
    exit()

print("[INFO] Full Hybrid Tracking dimulai...")

# Loop Utama
prev_time = time.time()
last_pwm_send = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    frame_h, frame_w = frame.shape[:2]

    # YOLO deteksi orang
    results = model(frame, imgsz=320, verbose=False, device=device, half=True)[0]
    detections = []
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if cls == 0 and conf > 0.5:  # Class Person
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

    # DeepSort update
    tracks = tracker.update_tracks(detections, frame=frame)
    face_center = None

    for track in tracks:
        if not track.is_confirmed():
            continue
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_w, x2), min(frame_h, y2)
        roi_person = frame[y1:y2, x1:x2]
        if roi_person is None or roi_person.size == 0:
            continue

        gray_roi = cv2.cvtColor(roi_person, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_roi, 1.1, 5, minSize=(50, 50))

        if len(faces) > 0:
            (fx, fy, fw, fh) = max(faces, key=lambda f: f[2]*f[3])
            abs_x1, abs_y1 = x1 + fx, y1 + fy
            abs_x2, abs_y2 = abs_x1 + fw, abs_y1 + fh
            face_crop = roi_person[fy:fy+fh, fx:fx+fw]
            if face_crop.size > 0:
                face_to_recognize = face_crop.copy()

            with recognition_lock:
                display_name = recognized_name
                conf_score = recognized_conf

            color = (0, 255, 0) if display_name != "Unknown" else (0, 0, 255)
            label_text = f"{display_name} ({conf_score:.1f}) [{track.track_id}]"
            cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), color, 2)
            cv2.putText(frame, label_text, (abs_x1, abs_y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cx_new = (abs_x1 + abs_x2) // 2
            cy_new = (abs_y1 + abs_y2) // 2

            # Filter halus posisi wajah agar tidak melompat
            if last_face_cx is not None and last_face_cy is not None:
                cx_new = int(last_face_cx * (1 - FACE_SMOOTH_FACTOR) + cx_new * FACE_SMOOTH_FACTOR)
                cy_new = int(last_face_cy * (1 - FACE_SMOOTH_FACTOR) + cy_new * FACE_SMOOTH_FACTOR)

            last_face_cx, last_face_cy = cx_new, cy_new
            face_center = (cx_new, cy_new)
            last_face_time = time.time()
            status_text = "TRACKING"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)

    # Kontrol Servo
    if face_center is not None:
        pwm_x, pwm_y = calculate_pwm_target(face_center[0], face_center[1], frame_w, frame_h, deadzone=DEADZONE)
        if pwm_x is not None:
            servo_x = (1 - SMOOTH_FACTOR) * servo_x + SMOOTH_FACTOR * pwm_x
            servo_y = (1 - SMOOTH_FACTOR) * servo_y + SMOOTH_FACTOR * pwm_y
            if (time.time() - last_pwm_send) > 0.2:
                send_pwm_async(servo_x, servo_y)
                last_pwm_send = time.time()

    elif (time.time() - last_face_time) > AUTO_CENTER_DELAY:
        dist_to_center = abs(servo_x - 90) + abs(servo_y - 90)
        if dist_to_center > 2:
            status_text = "AUTO-CENTERING"
            servo_x = (1 - SMOOTH_FACTOR) * servo_x + SMOOTH_FACTOR * 90
            servo_y = (1 - SMOOTH_FACTOR) * servo_y + SMOOTH_FACTOR * 90
            if (time.time() - last_pwm_send) > 0.5:
                send_pwm_async(servo_x, servo_y)
                last_pwm_send = time.time()
        else:
            status_text = "WAITING"
    else:
        status_text = "LOST"

    # Gambar deadzone
    cv2.rectangle(frame,
                  (frame_w//2 - DEADZONE, frame_h//2 - DEADZONE),
                  (frame_w//2 + DEADZONE, frame_h//2 + DEADZONE),
                  (255, 255, 255), 1)

    # Info Overlay
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent
    fps = 1.0 / (time.time() - prev_time + 1e-6)
    prev_time = time.time()

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"CPU:{cpu:.0f}% MEM:{mem:.0f}%", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"STATUS: {status_text}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0) if status_text == "TRACKING" else ((0, 255, 255) if status_text == "LOST" else (255, 255, 0)), 2)

    cv2.imshow("Hybrid YOLO + DeepSort + Haar + LBPH + Deadzone + Smooth", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()