# hardware/camera.py
import cv2
from utils.logger import get_logger

class Camera:
    def __init__(self, device, width, height, fps):
        self.logger = get_logger("Camera")
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        self.configure(width, height, fps)

    def configure(self, width, height, fps):
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.logger.info(f"Camera configured: {width}x{height}@{fps}fps")

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            self.logger.error("Failed to read frame")
            return None
        return frame

    def release(self):
        self.cap.release()