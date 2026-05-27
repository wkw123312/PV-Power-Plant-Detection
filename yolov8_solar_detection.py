"""
光伏电站遥感识别与面积估算脚本
"""
import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from math import cos, radians
from config import load_config

config = load_config()

# 模型训练脚本
def train_yolov8(model_path=None, data_path=None, imgsz=640, epochs=100):
    model_path = model_path or config.get("model_path")
    data_path = data_path or config.get("dataset_yaml")
    """
    使用 YOLOv8 进行模型训练
    :param model_path: 预训练权重路径
    :param data_path: 数据集路径，包含 images 和 labels 文件夹
    :param imgsz: 输入图像大小
    :param epochs: 训练轮数
    """
    print("加载 YOLOv8 模型...")
    model = YOLO(model_path)  # 加载 YOLOv8-Seg 预训练模型
    print("开始训练...")
    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        augment=True  # 数据增强
    )
    print("模型训练完成！")

# 面积计算核心算法
def calculate_area(pixel_count, gsd, tilt_angle):
    """
    根据像素数量和地面采样间隔计算实际面积
    :param pixel_count: 分割区域的像素数量
    :param gsd: 地面采样间隔（单位：米）
    :param tilt_angle: 光伏组件倾角（单位：度）
    :return: 投影面积和实际面积
    """
    projected_area = pixel_count * (gsd ** 2)  # 投影面积
    actual_area = projected_area / cos(radians(tilt_angle))  # 实际面积
    return projected_area, actual_area

# 推理与面积估算
def infer_and_calculate(model_path, test_images_path, gsd, tilt_angle, output_csv):
    """
    对测试图像进行推理并计算面积
    :param model_path: 训练好的模型路径
    :param test_images_path: 测试图像文件夹路径
    :param gsd: 地面采样间隔
    :param tilt_angle: 光伏组件倾角
    :param output_csv: 输出结果的 CSV 文件路径
    """
    model = YOLO(model_path)  # 加载训练好的模型
    results = []

    for image_name in os.listdir(test_images_path):
        image_path = os.path.join(test_images_path, image_name)
        image = cv2.imread(image_path)

        # 推理
        result = model(image)
        masks = result[0].masks.data.cpu().numpy()  # 获取分割掩码
        pixel_count = np.sum(masks > 0.5)  # 计算像素数量

        # 面积计算
        projected_area, actual_area = calculate_area(pixel_count, gsd, tilt_angle)

        # 可视化
        annotated_image = result[0].plot()  # 绘制检测结果
        annotated_image = cv2.putText(
            annotated_image,
            f"Area: {actual_area:.2f} m^2",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        output_image_path = os.path.join(test_images_path, f"annotated_{image_name}")
        cv2.imwrite(output_image_path, annotated_image)

        # 保存结果
        results.append({
            "文件名": image_name,
            "检测数量": len(result[0].boxes),
            "总投影面积": projected_area,
            "总实际面积": actual_area
        })

    # 导出结果到 CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"推理完成，结果已保存到 {output_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="光伏电站遥感识别与面积估算")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "infer"], help="运行模式：train 或 infer")
    parser.add_argument("--data", type=str, default=None, help="数据集路径，仅在 train 模式下需要")
    parser.add_argument("--model", type=str, default=None, help="预训练权重路径，仅在 train 模式下需要")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像大小，仅在 train 模式下需要")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数，仅在 train 模式下需要")
    parser.add_argument("--test", type=str, help="测试图像文件夹路径，仅在 infer 模式下需要")
    parser.add_argument("--gsd", type=float, help="地面采样间隔，仅在 infer 模式下需要")
    parser.add_argument("--tilt", type=float, help="光伏组件倾角，仅在 infer 模式下需要")
    parser.add_argument("--output", type=str, help="输出结果的 CSV 文件路径，仅在 infer 模式下需要")
    args = parser.parse_args()

    if args.config:
        global config
        config = load_config(args.config)

    if args.mode == "train":
        train_yolov8(args.model, args.data, args.imgsz, args.epochs)
    elif args.mode == "infer":
        infer_and_calculate(args.model or config.get("model_path"), args.test, args.gsd, args.tilt, args.output)
