import cv2
import numpy as np
import os
from pathlib import Path

# 定义输入和输出文件夹
input_folder = 'Output'  # 存放原始图片的文件夹
output_folder = 'out'  # 存放增强后图片的文件夹

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 创建CLAHE对象
clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))

# 递归处理文件夹中的所有图片
def process_images(input_dir, output_dir):
    # 获取输入文件夹中的所有图片文件
    for item in os.listdir(input_dir):
        input_path = os.path.join(input_dir, item)
        
        # 如果是文件夹，递归处理
        if os.path.isdir(input_path):
            # 创建对应的输出子文件夹
            sub_output_dir = os.path.join(output_dir, item)
            os.makedirs(sub_output_dir, exist_ok=True)
            process_images(input_path, sub_output_dir)
        
        # 如果是图片文件，进行处理
        elif item.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            # 读取图片
            image = cv2.imread(input_path)
            
            if image is None:
                print(f"无法读取图片: {input_path}")
                continue
            
            # 转换为灰度图
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 应用CLAHE
            clahe_result = clahe.apply(gray_image)
            
            # 构建输出文件名和路径
            filename = Path(item).stem
            output_path = os.path.join(output_dir, f"{filename}.jpg")
            
            # 保存处理后的图片
            cv2.imwrite(output_path, clahe_result)
            print(f"已处理并保存: {output_path}")

# 开始处理图片
print("开始处理图片...")
process_images(input_folder, output_folder)
print("所有图片处理完成")
