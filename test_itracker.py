# test_itracker.py
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from ITrackerModel import ITrackerModel
from detector import detect_face_and_eyes
from utils import generate_face_grid

# === 설정 ===
SCREEN_W, SCREEN_H = 1920, 1080
# MODEL_PATH = "itracker_trained_1000_256_0.0001.pth"
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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # 왼쪽-오른쪽 반전
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

            # 시선 위치 정규화
            gaze_x = int(np.clip(gaze_x, 0, SCREEN_W-1))
            gaze_y = int(np.clip(gaze_y, 0, SCREEN_H-1))

            cv2.circle(input_frame, (gaze_x, gaze_y), 5, (0, 255, 0), -1)
            cv2.putText(input_frame, f"Gaze: ({gaze_x}, {gaze_y})", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

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
