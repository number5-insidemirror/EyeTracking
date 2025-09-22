import os
import random
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from ITrackerModel import ITrackerModel
import torchvision.transforms as transforms

# 설정
DATA_CSV = "dataset4/metadata.csv"
MODEL_PATH = "1000_0005_128_nonorm.pth"
NUM_SAMPLES = 1200
VISUALIZE_NUM = 50
SCREEN_W, SCREEN_H = 1920, 1080
VISUALIZE_DIR = "result100_50"
ACCURACY_THRESHOLD = 100  # 정확도 기준 오차

os.makedirs(VISUALIZE_DIR, exist_ok=True)

# 전처리 함수
def preprocess(img_path):
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(img)

# 모델 로딩
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ITrackerModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# 데이터 로딩 및 샘플링
df = pd.read_csv(DATA_CSV)
df = df.sample(n=min(NUM_SAMPLES, len(df)), random_state=50).reset_index(drop=True)

errors = []
correct_count = 0
vis_indices = set(random.sample(range(len(df)), min(VISUALIZE_NUM, len(df))))

for i, row in df.iterrows():
    try:
        face = preprocess(row['face_path']).unsqueeze(0).to(device)
        eye_left = preprocess(row['left_eye_path']).unsqueeze(0).to(device)
        eye_right = preprocess(row['right_eye_path']).unsqueeze(0).to(device)

        face_grid = np.load(row['face_grid_path'])  # shape: (25, 25)
        face_grid = torch.tensor(face_grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(face, eye_left, eye_right, face_grid)

        pred_x, pred_y = pred[0].cpu().numpy()
        pred_x = np.clip(pred_x, 0, SCREEN_W - 1)
        pred_y = np.clip(pred_y, 0, SCREEN_H - 1)

        true_x, true_y = row['gaze_x'], row['gaze_y']
        error = np.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)
        errors.append(error)

        if error <= ACCURACY_THRESHOLD:
            correct_count += 1

        if i in vis_indices:
            # 흰 배경 생성
            white_canvas = np.ones((SCREEN_H, SCREEN_W, 3), dtype=np.uint8) * 255

            center = (int(true_x), int(true_y))

            # (1) 투명한 파란 원 그리기 (반경 100px, alpha blending)
            overlay = white_canvas.copy()
            cv2.circle(overlay, center, ACCURACY_THRESHOLD, (255, 0, 0), -1)  # 파란 원
            alpha = 0.3  # 투명도 30%
            cv2.addWeighted(overlay, alpha, white_canvas, 1 - alpha, 0, white_canvas)

            # (2) 정답 및 예측 좌표 점 그리기
            cv2.circle(white_canvas, center, 20, (0, 0, 255), -1)  # 정답: 빨간색
            cv2.circle(white_canvas, (int(pred_x), int(pred_y)), 20, (0, 255, 0), -1)  # 예측: 초록색

            # (3) 좌표 텍스트
            cv2.putText(white_canvas, f"GT: ({int(true_x)}, {int(true_y)})", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
            cv2.putText(white_canvas, f"Pred: ({int(pred_x)}, {int(pred_y)})", (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)

            # (4) 정답 여부 텍스트 (좌측 상단)
            correctness_text = "Correct" if error <= ACCURACY_THRESHOLD else "Incorrect"
            color = (0, 200, 0) if correctness_text == "Correct" else (0, 0, 200)
            cv2.putText(white_canvas, correctness_text, (30, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)

            # 저장
            save_path = os.path.join(VISUALIZE_DIR, f"result_{i}.jpg")
            cv2.imwrite(save_path, white_canvas)


    except Exception as e:
        print(f"[{i}] 오류 발생: {e}")
        continue

# 출력
if errors:
    avg_error = np.mean(errors)
    accuracy = (correct_count / len(errors)) * 100
    print(f"평가 완료: {len(errors)}개 샘플")
    print(f"평균 오차: {avg_error:.2f} pixels")
    print(f"정확도 (오차 {ACCURACY_THRESHOLD}px 이하): {accuracy:.2f}%")
else:
    print("유효한 이미지가 없습니다.")
