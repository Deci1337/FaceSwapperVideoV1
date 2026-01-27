import mediapipe as mp
import cv2
import numpy as np
from typing import Optional

class LandmarksVisualizer:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        
    def draw_hands(self, image: np.ndarray, multi_hand_landmarks):
        if not multi_hand_landmarks:
            return
            
        for hand_landmarks in multi_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style())
                
    def draw_face_mesh(self, image: np.ndarray, multi_face_landmarks):
        if not multi_face_landmarks:
            return
            
        for face_landmarks in multi_face_landmarks:
            self.mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
                
            self.mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles
                .get_default_face_mesh_contours_style())

    def draw_face_bbox(self, image: np.ndarray, multi_face_landmarks):
        if not multi_face_landmarks:
            return
            
        h, w, _ = image.shape
        for face_landmarks in multi_face_landmarks:
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x < x_min: x_min = x
                if x > x_max: x_max = x
                if y < y_min: y_min = y
                if y > y_max: y_max = y
                
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

