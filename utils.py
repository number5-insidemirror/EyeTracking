# utils.py
import numpy as np
import cv2
import os
from datetime import datetime
from PIL import Image

def generate_face_grid(bbox, frame_shape, grid_size=25):
    """
    얼굴 bounding box 기준으로 face grid 생성
    """
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    h, w = frame_shape[:2]
    cell_h, cell_w = h / grid_size, w / grid_size

    x1 = int(bbox.left() / cell_w)
    y1 = int(bbox.top() / cell_h)
    x2 = int(bbox.right() / cell_w)
    y2 = int(bbox.bottom() / cell_h)

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(grid_size - 1, x2), min(grid_size - 1, y2)
    grid[y1:y2+1, x1:x2+1] = 1.0

    return grid

def save_image(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(path)

def save_numpy_array(array, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, array)

def save_metadata(metadata_path, entry):
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    if not os.path.exists(metadata_path):
        with open(metadata_path, 'w') as f:
            f.write("timestamp,face_path,left_eye_path,right_eye_path,face_grid_path,gaze_x,gaze_y\n")

    with open(metadata_path, 'a') as f:
        f.write(','.join(map(str, entry)) + '\n')

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")
