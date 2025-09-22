import os
import cv2
import time
from detector import detect_face_and_eyes
from utils import generate_face_grid, save_image, save_numpy_array, save_metadata, get_timestamp
import numpy as np
# === 설정 ===
GRID_W, GRID_H = 30, 15
SCREEN_W, SCREEN_H = 1920, 1080
POINT_RADIUS = 10
POINT_COLOR = (0, 0, 255)

# === 5x5 균일 대표 점 선택 ===
sample_x = np.linspace(0, GRID_W - 1, 5, dtype=int)
sample_y = np.linspace(0, GRID_H - 1, 5, dtype=int)

gaze_points = [
    (int((i + 0.5) * SCREEN_W / GRID_W), int((j + 0.5) * SCREEN_H / GRID_H))
    for j in sample_y for i in sample_x
]

# === 저장 경로 ===
BASE_PATH = "dataset_alpha"
FACE_PATH = f"{BASE_PATH}/face_images"
LEFT_EYE_PATH = f"{BASE_PATH}/eye_left"
RIGHT_EYE_PATH = f"{BASE_PATH}/eye_right"
GRID_PATH = f"{BASE_PATH}/face_grid"
META_PATH = f"{BASE_PATH}/metadata.csv"

os.makedirs(FACE_PATH, exist_ok=True)
os.makedirs(LEFT_EYE_PATH, exist_ok=True)
os.makedirs(RIGHT_EYE_PATH, exist_ok=True)
os.makedirs(GRID_PATH, exist_ok=True)

# === 카메라 설정 ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_H)

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
    cv2.putText(display, f"Point {current_idx + 1}/{len(gaze_points)}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imshow("Gaze Collector", display)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord(' '):
        print(f"\n▶ {current_idx+1}번 점 수집 시작 (총 3초, 앞 1초 제외하고 2초간 저장)...")
        start_time = time.time()
        gaze_x, gaze_y = point

        while time.time() - start_time < 2.0:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            display = frame.copy()
            cv2.circle(display, point, POINT_RADIUS, POINT_COLOR, -1)
            cv2.putText(display, f"Point {current_idx + 1}/{len(gaze_points)}, Capturing...",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            elapsed = time.time() - start_time
            if elapsed >= 1.0:  # 저장은 1초 지난 후부터
                result = detect_face_and_eyes(frame)
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

                    print(f"✅ 저장됨: {timestamp}, 좌표=({gaze_x}, {gaze_y})")

            cv2.imshow("Gaze Collector", display)
            cv2.waitKey(100)  # 10fps

        current_idx += 1

cap.release()
cv2.destroyAllWindows()
