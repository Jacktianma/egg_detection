import os
import random
import shutil
from pathlib import Path

def split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # 检查比例之和是否近似为 1
    epsilon = 1e-9
    assert abs(train_ratio + val_ratio + test_ratio - 1) < epsilon, "The sum of ratios must be approximately 1."

    # 获取所有图片文件
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.seed(42) 
    random.shuffle(image_files)

    # 计算划分的索引
    num_images = len(image_files)
    train_index = int(num_images * train_ratio)
    val_index = train_index + int(num_images * val_ratio)

    # 划分数据集
    train_files = image_files[:train_index]
    val_files = image_files[train_index:val_index]
    test_files = image_files[val_index:]

    # 创建输出文件夹
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_images_dir = output_dir / 'images' / 'train'
    train_labels_dir = output_dir / 'labels' / 'train'
    val_images_dir = output_dir / 'images' / 'val'
    val_labels_dir = output_dir / 'labels' / 'val'
    test_images_dir = output_dir / 'images' / 'test'
    test_labels_dir = output_dir / 'labels' / 'test'

    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir, test_images_dir, test_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # 移动训练集文件
    for file in train_files:
        image_path = Path(images_dir) / file
        label_path = Path(labels_dir) / (os.path.splitext(file)[0] + '.txt')
        shutil.copy(image_path, train_images_dir)
        shutil.copy(label_path, train_labels_dir)

    # 移动验证集文件
    for file in val_files:
        image_path = Path(images_dir) / file
        label_path = Path(labels_dir) / (os.path.splitext(file)[0] + '.txt')
        shutil.copy(image_path, val_images_dir)
        shutil.copy(label_path, val_labels_dir)

    # 移动测试集文件
    for file in test_files:
        image_path = Path(images_dir) / file
        label_path = Path(labels_dir) / (os.path.splitext(file)[0] + '.txt')
        shutil.copy(image_path, test_images_dir)
        shutil.copy(label_path, test_labels_dir)

    print("Dataset split completed.")

if __name__ == "__main__":
    # 替换为你的图片文件夹路径
    images_dir = "photo\photo"  
    # 替换为你的标签文件夹路径
    labels_dir = "photo\labels"
    # 替换为输出文件夹路径
    output_dir = "Output"
    # 可以根据需要调整比例
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    split_dataset(images_dir, labels_dir, output_dir, train_ratio, val_ratio, test_ratio)