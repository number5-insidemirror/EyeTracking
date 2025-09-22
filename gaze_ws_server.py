import asyncio
import websockets
import json
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from ITrackerModel import ITrackerModel
from detector import detect_face_and_eyes
from utils import generate_face_grid

# === 설정 ===
SCREEN_W, SCREEN_H = 1920, 1080
MODEL_PATH = "1000_0005_128_nonorm.pth"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

async def send_gaze(websocket):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ITrackerModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_H)

    print("✅ WebSocket Gaze Server Started at ws://localhost:3000")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            input_frame = frame.copy()

            result = detect_face_and_eyes(frame)
            if result is not None:
                face_img, left_eye, right_eye, bbox = result
                face_grid = generate_face_grid(bbox, frame.shape)

                face = transform(Image.fromarray(face_img)).unsqueeze(0).to(device)
                eyeL = transform(Image.fromarray(left_eye)).unsqueeze(0).to(device)
                eyeR = transform(Image.fromarray(right_eye)).unsqueeze(0).to(device)
                grid = torch.tensor(face_grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

                with torch.no_grad():
                    pred = model(face, eyeL, eyeR, grid)

                gaze_x, gaze_y = pred[0].cpu().numpy()

                gaze_x = int(np.clip(gaze_x, 0, SCREEN_W - 1))
                gaze_y = int(np.clip(gaze_y, 0, SCREEN_H - 1))

                # 전송
                await websocket.send(json.dumps({"x": gaze_x, "y": gaze_y}))

            await asyncio.sleep(1 / 30)  # 30 FPS

    finally:
        cap.release()
        cv2.destroyAllWindows()

async def main():
    async with websockets.serve(send_gaze, "localhost", 8000):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
