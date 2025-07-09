import onnxruntime as ort
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------- STEP 1: 加载训练数据并拟合 scaler ----------
train_X = pd.read_csv("train_data.csv").values
scaler = StandardScaler()
scaler.fit(train_X)

# ---------- STEP 2: 输入待预测的样本 ----------
sample = [3,20,47,3,29,7,15,0,26,0,3,0,0,0,0,0,0,35,0,0,46,2,23,50]
if len(sample) != 24:
    raise ValueError("样本必须是长度为 24 的特征列表。")

# ---------- STEP 3: 标准化 ----------
X_input = scaler.transform([sample]).astype(np.float32)

# ---------- STEP 4: 加载 ONNX 模型 ----------
session = ort.InferenceSession("egg_fresh_mlp.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# ---------- STEP 5: 推理 ----------
logits = session.run([output_name], {input_name: X_input})[0]  # shape (1, 2)
probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)  # softmax
label = np.argmax(probabilities, axis=1)[0]

# ---------- STEP 6: 输出结果 ----------
label_map = {0: "好蛋", 1: "坏蛋"}
confidence = probabilities[0][label]
print(f"预测结果：{label_map[label]}，置信度：{confidence:.4f}")

# 可选：查看所有类别的概率
for i, prob in enumerate(probabilities[0]):
    print(f"类别 {label_map[i]} 的概率：{prob:.4f}")
