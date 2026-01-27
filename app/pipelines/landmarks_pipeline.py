import cv2
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
from app.config import LandmarksConfig
from app.io.video import VideoReader, VideoWriter
from app.utils.visualization import LandmarksVisualizer

class LandmarksPipeline:
    def __init__(self, config: LandmarksConfig):
        self.config = config
        self.reader = VideoReader(config.input_path)
        self.writer = VideoWriter(
            config.output_path, 
            fps=config.fps if config.fps else self.reader.fps,
            resolution=(self.reader.width, self.reader.height)
        )
        self.viz = LandmarksVisualizer()
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def run(self):
        print(f"Processing landmarks: {self.config.input_path} -> {self.config.output_path}")
        
        for ret, frame in tqdm(self.reader.stream(), total=self.reader.frame_count):
            if not ret:
                break
                
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            
            # Process Hands
            if self.config.draw_hands:
                hands_results = self.mp_hands.process(frame_rgb)
                self.viz.draw_hands(frame, hands_results.multi_hand_landmarks)
                
            # Process Face
            face_results = self.mp_face_mesh.process(frame_rgb)
            if self.config.face_mode == 'mesh':
                self.viz.draw_face_mesh(frame, face_results.multi_face_landmarks)
            else:
                self.viz.draw_face_bbox(frame, face_results.multi_face_landmarks)
                
            self.writer.write(frame)
            
        self.reader.release()
        self.writer.release()
        self.mp_hands.close()
        self.mp_face_mesh.close()

