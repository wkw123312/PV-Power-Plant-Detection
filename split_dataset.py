import os
import cv2
import shutil
import random
import numpy as np
from tqdm import tqdm
from config import load_config

config = load_config()
BASE_DIR = os.path.abspath('.')
JPG_DIR = os.path.abspath(config.get("jpg_preview_dir", "./jpg_preview"))
PNG_DIR = os.path.abspath(config.get("mask_dir", "./label"))
OUTPUT_ROOT = os.path.abspath(config.get("dataset_root", "./datasets"))
TRAIN_RATIO = config.get("train_ratio", 0.8)

def mask_to_yolo_polygons(png_path, txt_path):
    """将非 0 像素值（如你的 43）转换为 YOLO 归一化坐标"""
    mask = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if mask is None or np.max(mask) == 0:
        return False

    # 只要像素值 > 0 就视为光伏板
    _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    # 注意这里是 findContours (大写 C，复数 s)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = mask.shape
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 5: continue
        points = cnt.reshape(-1, 2)
        norm_points = [f"{x/w:.6f} {y/h:.6f}" for x, y in points]
        polygons.append(f"0 {' '.join(norm_points)}")

    if polygons:
        with open(txt_path, 'w') as f:
            f.write('\n'.join(polygons))
        return True
    return False

def main():
    # 1. 清理并创建 YOLO 标准目录
    for part in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_ROOT, 'images', part), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_ROOT, 'labels', part), exist_ok=True)

    # 2. 获取所有文件（以 JPG 为准进行匹配）
    all_names = [f[:-4] for f in os.listdir(JPG_DIR) if f.endswith('.jpg')]
    random.seed(42) # 固定随机种子，方便复现
    random.shuffle(all_names)

    split_idx = int(len(all_names) * TRAIN_RATIO)
    train_set = all_names[:split_idx]
    val_set = all_names[split_idx:]

    print(f"📊 准备划分: 训练集 {len(train_set)}, 验证集 {len(val_set)}")

    # 3. 执行划分与转换
    for part, names in [('train', train_set), ('val', val_set)]:
        print(f"正在处理 {part} 数据...")
        count = 0
        for name in tqdm(names):
            src_jpg = os.path.join(JPG_DIR, name + ".jpg")
            src_png = os.path.join(PNG_DIR, name + ".png") # 对应 solar_2023_us_xxxx.png

            dst_jpg = os.path.join(OUTPUT_ROOT, 'images', part, name + ".jpg")
            dst_txt = os.path.join(OUTPUT_ROOT, 'labels', part, name + ".txt")

            if os.path.exists(src_png):
                if mask_to_yolo_polygons(src_png, dst_txt):
                    shutil.copy(src_jpg, dst_jpg)
                    count += 1
        print(f"✅ {part} 集完成，有效样本数: {count}")

if __name__ == "__main__":
    main()
