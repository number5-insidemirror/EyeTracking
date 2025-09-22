# detector.py
import dlib
import cv2
import numpy as np

# dlib 모델 로드 (얼굴 검출기 + 랜드마크 예측기)
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # 모델 필요

def detect_face_and_eyes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    if len(faces) == 0:
        return None

    face = faces[0]  # 첫 번째 얼굴만 사용
    shape = landmark_predictor(gray, face)

    # 눈 영역 좌표 (left eye: 36~41, right eye: 42~47)
    eye_left_pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(36, 42)])
    eye_right_pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(42, 48)])

    def crop_eye(pts):
        x, y, w, h = cv2.boundingRect(pts)
        cx, cy = x + w // 2, y + h // 2
        size = max(w, h) * 2
        x1, y1 = cx - size // 2, cy - size // 2
        x2, y2 = cx + size // 2, cy + size // 2
        eye_img = frame[max(0, y1):y2, max(0, x1):x2]
        return cv2.resize(eye_img, (224, 224))

    face_img = frame[face.top():face.bottom(), face.left():face.right()]
    face_img = cv2.resize(face_img, (224, 224))

    left_eye = crop_eye(eye_left_pts)
    right_eye = crop_eye(eye_right_pts)

    return face_img, left_eye, right_eye, face
