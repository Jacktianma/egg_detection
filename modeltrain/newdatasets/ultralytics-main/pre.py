from ultralytics import YOLO
import cv2
import time
import warnings
import numpy as np

# 忽略警告信息
warnings.filterwarnings("ignore", category=UserWarning)

def preprocess_frame(frame):
    """
    对摄像头帧进行与训练数据相同的预处理
    """
    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 应用CLAHE增强对比度
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 转回RGB格式 (与训练数据保持一致)
    rgb_frame = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    return rgb_frame

def main():
    # 加载模型 - 使用训练好的模型权重
    model = YOLO('D:/Acompetition/2025IC/目标灰度/runs/train/exp/weights/best.pt')
    print("模型已加载")
    
    # 打开摄像头 (0表示默认摄像头)
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("无法打开摄像头") 
        return
    
    # 计时器初始化
    prev_time = 0
    
    print("开始实时推理，按'q'键退出")
    
    while True:
        # 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            print("无法获取画面")
            break
        
        # 预处理帧 - 应用与训练数据相同的处理
        processed_frame = preprocess_frame(frame)
        
        # 计算FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        
        # 使用模型进行推理
        results = model(processed_frame, imgsz=320, conf=0.5, iou=0.1)
        
        # 在原始帧上绘制结果
        annotated_frame = results[0].plot()
        
        # 添加FPS信息
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示结果
        cv2.imshow("YOLOv8 实时检测", annotated_frame)
        
        # # 可选：显示预处理后的帧，以便调试
        # cv2.imshow("预处理后的帧", processed_frame)
        
        # 按'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    
    print("推理结束")

if __name__ == "__main__":
    main()
