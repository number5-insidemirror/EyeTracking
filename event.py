import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from ITrackerModel import ITrackerModel
from detector import detect_face_and_eyes
from utils import generate_face_grid
import time

# === 설정 ===
SCREEN_W, SCREEN_H = 1920, 1080
MODEL_PATH = "1000_0005_128_nonorm.pth"

def preprocess(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ITrackerModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_H)

    # 전체화면 창 설정
    cv2.namedWindow("Gaze Tracker - Press ESC to exit", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Gaze Tracker - Press ESC to exit", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # === 버튼 정의 (화면 90% 차지) ===
    btn_width = int(SCREEN_W * 0.9)
    btn_height = int(SCREEN_H * 0.9)
    btn_x = (SCREEN_W - btn_width) // 2
    btn_y = (SCREEN_H - btn_height) // 2
    button_rect = (btn_x, btn_y, btn_width, btn_height)

    # === 게이지 관련 변수 ===
    gaze_hold_time = 2.0
    dwell_start_time = None
    progress_ratio = 0.0
    clicked = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        input_frame = frame.copy()

        result = detect_face_and_eyes(frame)
        if result is not None:
            face_img, left_eye, right_eye, bbox = result
            face_grid = generate_face_grid(bbox, frame.shape)

            face = preprocess(face_img).unsqueeze(0).to(device)
            eyeL = preprocess(left_eye).unsqueeze(0).to(device)
            eyeR = preprocess(right_eye).unsqueeze(0).to(device)
            grid = torch.tensor(face_grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(face, eyeL, eyeR, grid)

            gaze_x, gaze_y = pred[0].cpu().numpy()
            gaze_x = int(np.clip(gaze_x, 0, SCREEN_W - 1))
            gaze_y = int(np.clip(gaze_y, 0, SCREEN_H - 1))

            cv2.circle(input_frame, (gaze_x, gaze_y), 5, (0, 255, 0), -1)
            cv2.putText(input_frame, f"Gaze: ({gaze_x}, {gaze_y})", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # === 버튼 처리 ===
            bx, by, bw, bh = button_rect
            in_button = bx <= gaze_x <= bx + bw and by <= gaze_y <= by + bh
            current_time = time.time()

            if in_button:
                if dwell_start_time is None:
                    dwell_start_time = current_time
                elapsed = current_time - dwell_start_time
                progress_ratio = min(elapsed / gaze_hold_time, 1.0)

                # 원형 게이지
                center = (bx + bw // 2, by + bh // 2)
                radius = 60
                thickness = 8
                angle = int(360 * progress_ratio)
                cv2.ellipse(input_frame, center, (radius, radius), -90, 0, angle, (0, 255, 255), thickness)

                if progress_ratio >= 1.0 and not clicked:
                    print("✅ 버튼 클릭됨!")
                    cv2.putText(input_frame, "Clicked!", (bx + 50, by - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    clicked = True
            else:
                dwell_start_time = None
                progress_ratio = 0.0
                clicked = False

            # 버튼 그리기
            cv2.rectangle(input_frame, (bx, by), (bx + bw, by + bh), (0, 128, 255), 3)
            cv2.putText(input_frame, "BIG BUTTON", (bx + 40, by + bh // 2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        else:
            cv2.putText(input_frame, "Face not detected", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow("Gaze Tracker - Press ESC to exit", input_frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
