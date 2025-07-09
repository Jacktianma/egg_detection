from ultralytics import YOLO

def export_model():
    # 加载训练好的模型
    model = YOLO('D:/Acompetition/2025IC/目标灰度/runs/train/exp/weights/best.pt')

    # 导出模型为 ONNX 格式
    results = model.export(format='onnx', imgsz=320)

    if results:
        print("模型成功导出为 ONNX 格式。")
    else:
        print("模型导出失败。")

if __name__ == "__main__":
    export_model()