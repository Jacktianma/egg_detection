
import warnings
import os
import cv2
import numpy as np
from pathlib import Path

def convert_grayscale_to_rgb(input_dir, output_dir=None):
    """
    将指定目录下的灰度图像转换为RGB图像
    
    Args:
        input_dir: 输入目录路径，包含灰度图像
        output_dir: 输出目录路径，如果为None则会在input_dir旁创建一个新目录
    """
    if output_dir is None:
        # 如果未指定输出目录，则创建一个名为"原目录名_rgb"的新目录
        parent_dir = str(Path(input_dir).parent)
        dir_name = Path(input_dir).name
        output_dir = os.path.join(parent_dir, f"{dir_name}_rgb")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 递归处理文件夹中的所有图片
    def process_images(in_dir, out_dir):
        # 获取输入文件夹中的所有图片文件
        for item in os.listdir(in_dir):
            input_path = os.path.join(in_dir, item)
            
            # 如果是文件夹，递归处理
            if os.path.isdir(input_path):
                # 创建对应的输出子文件夹
                sub_output_dir = os.path.join(out_dir, item)
                os.makedirs(sub_output_dir, exist_ok=True)
                process_images(input_path, sub_output_dir)
            
            # 如果是图片文件，进行处理
            elif item.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                # 读取图片
                image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
                
                if image is None:
                    print(f"无法读取图片: {input_path}")
                    continue
                
                # 将灰度图转换为RGB图像
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                
                # 构建输出文件路径（保持原文件名）
                output_path = os.path.join(out_dir, item)
                
                # 保存处理后的图片
                cv2.imwrite(output_path, rgb_image)
                print(f"已转换并保存: {output_path}")
    
    # 开始处理图片
    print("开始将灰度图转换为RGB图像...")
    process_images(input_dir, output_dir)
    print("所有图片转换完成")
    
    return output_dir

# 设置图像目录
img_dir = 'out/images'  # 替换为您的图像目录

# 转换图像
rgb_img_dir = convert_grayscale_to_rgb(img_dir)

# 更新您的mydata.yaml文件中的路径指向新的RGB图像目录
# 这一步可能需要手动完成，或者您可以编写代码来修改YAML文件
