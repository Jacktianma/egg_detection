import ast
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 标签映射
label_map = {'5.txt': 0,'3.txt': 1}
file_paths = ['5.txt', '3.txt']

# 加载数据
samples, labels = [], []
for path in file_paths:
    label = label_map[path]
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = ast.literal_eval(line.strip())
                samples.append(obj)
                labels.append(label)
            except:
                print(f"格式错误：{line.strip()}")

print("样本类别分布：", Counter(labels))

# 特征处理
X_raw = pd.DataFrame(samples).values
y = np.array(labels)

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)
joblib.dump(scaler, "scaler.pkl")  # 保存标准化器

# 保存原始数据
pd.DataFrame(X_raw).to_csv("train_data.csv", index=False)

# 类别权重
weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
print(f"类别权重: {class_weights.cpu().numpy()}")

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 转换为 Tensor
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

# 模型定义
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=24, output_dim=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)

model = MLPClassifier().to(device)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练过程
epochs = 50
train_losses, test_losses = [], []
train_accs, test_accs = [], []

for epoch in range(epochs):
    model.train()
    correct, total, epoch_loss = 0, 0, 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * xb.size(0)
        correct += (preds.argmax(1) == yb).sum().item()
        total += yb.size(0)

    train_losses.append(epoch_loss / total)
    train_accs.append(correct / total)

    # 验证
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            val_loss += loss.item() * xb.size(0)
            val_correct += (preds.argmax(1) == yb).sum().item()
            val_total += yb.size(0)

    test_losses.append(val_loss / val_total)
    test_accs.append(val_correct / val_total)

    print(f"Epoch {epoch+1}/{epochs} | "
          f"Train Loss: {train_losses[-1]:.4f} | Val Loss: {test_losses[-1]:.4f} | "
          f"Train Acc: {train_accs[-1]:.4f} | Val Acc: {test_accs[-1]:.4f}")

# 评估
model.eval()
with torch.no_grad():
    preds = model(X_test.to(device))
    y_pred = preds.argmax(1).cpu().numpy()

print("\n分类报告：")
print(classification_report(y_test.numpy(), y_pred, target_names=["好蛋", "坏蛋"]))
print("混淆矩阵：")
print(confusion_matrix(y_test.numpy(), y_pred))


# 保存模型
torch.save(model.state_dict(), "egg_fresh_mlp.pth")
print("模型已保存为 egg_fresh_mlp.pth")

# ONNX 导出
model_cpu = model.to("cpu")
dummy_input = torch.randn(1, 24)
torch.onnx.export(
    model_cpu, dummy_input, "egg_fresh_mlp.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=11
)
print("模型已导出为 ONNX 格式")

# 可视化
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Val Loss")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label="Train Acc")
plt.plot(test_accs, label="Val Acc")
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("training_result_mlp.png")
plt.show()
