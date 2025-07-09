from ultralytics import YOLO
import cv2

clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))

def detect_from_camera():
    # 加载预训练的 YOLOv8 模型
    model = YOLO('runs/train/exp/weights/best.pt')

    # 打开摄像头
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        # 读取一帧视频
        ret, frame = cap.read()

        if not ret:
            print("无法读取帧")
            break
        # 创建 CLAHE 对象（可放外层避免重复创建）
        results = model(frame,iou=0.5, imgsz=320)
        

        # 遍历检测结果
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 获取类别索引
                cls = int(box.cls[0])
                # 获取类别名称
                class_name = model.names[cls]
                # 获取置信度
                conf = float(box.conf[0])
                # 获取边界框坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # 在终端打印检测信息
                print(f"检测到目标: {class_name}, 置信度: {conf:.2f}, 边界框: ({x1}, {y1}), ({x2}, {y2})")

        # 显示检测结果
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 实时检测", annotated_frame)

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_from_camera()