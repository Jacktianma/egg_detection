# models/freshness_model.py
import onnxruntime
import numpy as np
from utils.logger import get_logger

class FreshnessModel:
    def __init__(self, model_path):
        self.logger = get_logger("FreshnessModel")
        self.session = None
        self.load_model(model_path)
        self.mean = np.array([...])  # 从配置或常量中加载
        self.std = np.array([...])

    def load_model(self, model_path):
        try:
            self.session = onnxruntime.InferenceSession(model_path)
            self.logger.info("Freshness ONNX model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load ONNX model: {e}")
            raise

    def predict(self, sample_list):
        try:
            if len(sample_list) != 24:
                raise ValueError("Input data length must be 24")
            input_data = (np.array(sample_list) - self.mean) / (self.std + 1e-8)
            input_data = input_data.astype(np.float32).reshape(1, 24)
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name
            logits = self.session.run([output_name], {input_name: input_data})[0]
            probs = self.softmax(logits)
            freshness_score = float(probs[0, 0])
            return self.classify_freshness(freshness_score)
        except Exception as e:
            self.logger.error(f"Freshness prediction failed: {e}")
            raise

    def softmax(self, logits):
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def classify_freshness(self, score):
        if score >= 0.90:
            return "goodegg", f"极鲜({score:.2%})"
        elif score >= 0.75:
            return "goodegg", f"新鲜({score:.2%})"
        elif score >= 0.50:
            return "goodegg", f"次新鲜({score:.2%})"
        elif score >= 0.25:
            return "badegg", f"临界期({score:.2%})"
        else:
            return "badegg", f"不新鲜({score:.2%})"