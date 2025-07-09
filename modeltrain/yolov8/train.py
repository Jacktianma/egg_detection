from ultralytics import YOLO
import os

def train_yolov8():
    # 获取当前脚本路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml = os.path.join(current_dir, 'mydata.yaml')
    
    # 使用更强的预训练模型
    model = YOLO('yolov8n.pt')  # yolov8n.pt 可改为 yolov8s.pt 或 yolov8m.pt
    
    # 训练配置
    results = model.train(
        data=data_yaml,
        epochs=100,               # 训练轮数增加
        imgsz=320,                # 使用更高图像分辨率，便于检测细小裂纹
        batch=32,                 # batch根据你GPU调整，太大会爆显存
        device='0',
        workers=8,                # 降低数据加载压力
        patience=50,            # 增大早停容忍
        save=True,
        project='runs/train',
        name='crack_detect_expv8s',
        exist_ok=True
    )
    
    return results

if __name__ == '__main__':
    train_yolov8()
