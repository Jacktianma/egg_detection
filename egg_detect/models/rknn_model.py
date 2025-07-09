# models/rknn_model.py
from rknnlite.api import RKNNLite
import numpy as np
from utils.logger import get_logger

class RKNNModel:
    def __init__(self, model_path):
        self.logger = get_logger("RKNNModel")
        self.rknn_lite = RKNNLite()
        self.load_model(model_path)

    def load_model(self, model_path):
        self.logger.info("Loading RKNN model")
        if self.rknn_lite.load_rknn(model_path) != 0:
            raise RuntimeError("Failed to load RKNN model")
        if self.rknn_lite.init_runtime() != 0:
            raise RuntimeError("Failed to initialize runtime environment")
        self.logger.info("RKNN model initialized successfully")

    def infer(self, input_data):
        return self.rknn_lite.inference(inputs=[input_data])

    def release(self):
        self.rknn_lite.release()