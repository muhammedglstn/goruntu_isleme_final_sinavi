from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import os

def blur_face_by_landmarks(image, face_landmarks):
    h, w, _ = image.shape
    points = []
    for lm in face_landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        points.append([x, y])
    points = np.array(points, dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60))
    mask = cv2.dilate(mask, kernel, iterations=1)
    blurred = cv2.GaussianBlur(image, (99, 99), 30)
    face_blurred = np.where(mask[..., None] == 255, blurred, image)
    return face_blurred

def draw_landmarks_on_image_and_blur_face(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)
    for face_landmarks in face_landmarks_list:
        annotated_image = blur_face_by_landmarks(annotated_image, face_landmarks)
    return annotated_image

def sutun_basliklarini_olustur():
    with open("veriseti.csv", "w") as f:
        satir = ""
        for i in range(1, 479):
            satir += f"x{i},y{i},"
        satir += "Etiket\n"
        f.write(satir)

etiket = "happy"
sutun_basliklarini_olustur()

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
model_path = os.path.join(desktop_path, "face_landmarker_v2_with_blendshapes.task")

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

cam = cv2.VideoCapture(0)

while cam.isOpened():
    basari, frame = cam.read()
    if basari:
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = detector.detect(mp_image)
        annotated_image = draw_landmarks_on_image_and_blur_face(mp_image.numpy_view(), detection_result)
        cv2.imshow("yuz", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or cv2.getWindowProperty("yuz", cv2.WND_PROP_VISIBLE) < 1:
            break

cam.release()
cv2.destroyAllWindows()
