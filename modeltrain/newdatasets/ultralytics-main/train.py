from ultralytics import YOLO
import warnings
import os
import cv2
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
# 加载模型
model = YOLO('ultralytics-main/ultralytics/cfg/models/v8/yolov8n_swinTrans.yaml')

# 训练模型 - 使用转换后的RGB图像
if __name__ == '__main__':
    model.train(
        data='mydata.yaml',
        pretrained=False, 
        epochs=50,
        imgsz=320,
        device='0',
        batch=64,
        amp=True,
        workers=8,
        patience=20,             # 早停的耐心值
        project='runs/train',
        name='exp_swin',
        exist_ok=True
    )

