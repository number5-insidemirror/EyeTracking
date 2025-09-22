import cv2
import tkinter as tk
from detector import detect_face_and_eyes
from utils import *
import numpy as np
import time

# === 점 위치 설정 ===
GRID_W, GRID_H = 30, 15
SCREEN_W, SCREEN_H = 1920, 1080
POINT_RADIUS = 10
POINT_COLOR = (0, 0, 255)

gaze_points = [(int((i + 0.5) * SCREEN_W / GRID_W), int((j + 0.5) * SCREEN_H / GRID_H))
               for j in range(GRID_H) for i in range(GRID_W)]

# === 저장 경로 ===
BASE_PATH = "dataset2"
FACE_PATH = f"{BASE_PATH}/face_images"
LEFT_EYE_PATH = f"{BASE_PATH}/eye_left"
RIGHT_EYE_PATH = f"{BASE_PATH}/eye_right"
GRID_PATH = f"{BASE_PATH}/face_grid"
META_PATH = f"{BASE_PATH}/metadata.csv"

# === 카메라 설정 ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_H)

# === OpenCV 창 전체화면 설정 ===
cv2.namedWindow("Gaze Collector", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Gaze Collector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

current_idx = 0

while current_idx < len(gaze_points):
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    point = gaze_points[current_idx]
    display = frame.copy()
    cv2.circle(display, point, POINT_RADIUS, POINT_COLOR, -1)
    cv2.putText(display, f"point {current_idx + 1}/{len(gaze_points)}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.imshow("Gaze Collector", display)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

    elif key == ord(' '):
        print(f" 총 4초 수집 중, 앞 1초 제외하고 3초간 저장합니다... (point {current_idx + 1})")
        start_time = time.time()
        gaze_x, gaze_y = point

        while time.time() - start_time < 4.0:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            result = detect_face_and_eyes(frame)

            display = frame.copy()
            cv2.circle(display, point, POINT_RADIUS, POINT_COLOR, -1)
            cv2.putText(display, f"point {current_idx + 1}/{len(gaze_points)}, Capturing...",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            elapsed = time.time() - start_time

            # 1초 이상부터 저장 시작
            if elapsed >= 1.0:
                if result is not None:
                    face_img, left_eye, right_eye, bbox = result
                    face_grid = generate_face_grid(bbox, frame.shape)

                    timestamp = get_timestamp()
                    face_file = f"{FACE_PATH}/{timestamp}.jpg"
                    left_file = f"{LEFT_EYE_PATH}/{timestamp}.jpg"
                    right_file = f"{RIGHT_EYE_PATH}/{timestamp}.jpg"
                    grid_file = f"{GRID_PATH}/{timestamp}.npy"

                    save_image(face_img, face_file)
                    save_image(left_eye, left_file)
                    save_image(right_eye, right_file)
                    save_numpy_array(face_grid, grid_file)

                    save_metadata(META_PATH, [
                        timestamp, face_file, left_file, right_file,
                        grid_file, gaze_x, gaze_y
                    ])

                    print(f"⏱ 저장 시각: {elapsed:.2f}초 ✅ 저장됨")

            cv2.imshow("Gaze Collector", display)
            cv2.waitKey(100)  # 10fps

        current_idx += 1

cap.release()
cv2.destroyAllWindows()
# === 로그 저장 ===
from datetime import datetime
import os

log_dir = BASE_PATH
log_file = os.path.join(log_dir, "log.txt")

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open(log_file, 'a') as f:
    f.write(f"[{now}] 450 중 {current_idx}번째까지 수집됨\n")

print(f"📄 로그 저장 완료: {log_file}")


