import cv2
import numpy as np
import os
img_path = "D:/Acompetition/2025嵌赛/model/外壳检测/out/images_rgb/train/00001.jpg"
img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

print(img.shape)       # (H, W, C)  or  (H, W)

if img.ndim == 2:
    print("单通道 -> 灰度图")
elif img.shape[2] == 1:
    print("1 通道 -> 也是灰度图")
elif img.shape[2] == 3:
    print("3 通道 -> RGB/BGR 彩色图")
elif img.shape[2] == 4:
    print("4 通道 -> RGBA / BGRA（带透明度）")
