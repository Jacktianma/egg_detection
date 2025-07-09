from ultralytics import YOLO
import os

def train_yolov8():
    # 获取当前脚本所在的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml = os.path.join(current_dir, 'mydata.yaml')
    
    # 加载预训练模型
    model = YOLO('yolov8n.pt')
    
    # 开始训练，将 device 参数设置为 '0' 以使用第一块 GPU
    results = model.train(
        data=data_yaml,           # 数据配置文件路径
        epochs=50,                # 训练轮数
        imgsz=320,               # 图像大小
        batch=32,                 # 减小batch size
        device='0',            # 使用第一块GPU训练
        workers=8,               # 减少工作进程数
        patience=20,             # 早停的耐心值
        save=True,               # 保存模型
        project='runs/train',    # 保存结果的项目名称
        name='exp',              # 实验名称
        exist_ok=True           # 允许覆盖已存在的实验文件夹
    )
    
    return results

if __name__ == '__main__':
    train_yolov8()