# 光伏电站检测与识别

## Description

本项目为光伏电站遥感识别与面积估算工具，基于 YOLOv8 进行太阳能组件检测与分割。项目支持将公开的 GloSoFarID 数据集转换为训练格式、训练模型、以及对测试图像进行推理与面积计算。

## Dataset

数据集采用公开的 GloSoFarID 数据集（Global multispectral dataset for Solar Farm IDentification），包含多谱段卫星图像及对应的太阳能光伏板掩码标签。

## Installation

1. 克隆仓库并进入目录：
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd GloSoFarID-main
   ```
2. 创建并激活 Python 虚拟环境：
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
4. 安装 YAML 支持（如果需要）：
   ```bash
   pip install pyyaml
   ```

## Usage

1. 复制配置模板：
   ```bash
   cp config.yaml.example config.yaml
   ```
2. 编辑 `config.yaml`，将路径修改为你的本地目录，例如：
   - `dataset_root: ./datasets`
   - `dataset_yaml: ./dataset.yaml`
   - `model_path: ./yolov8n-seg.pt`
3. 运行数据集划分：
   ```bash
   python split_dataset.py
   ```
4. 训练模型：
   ```bash
   python yolov8_solar_detection.py --mode train --model ./yolov8n-seg.pt --data ./dataset.yaml
   ```
5. 推理与面积计算：
   ```bash
   python yolov8_solar_detection.py --mode infer --model ./runs/best.pt --test ./datasets/images/val --gsd 10 --tilt 0 --output ./results.csv
   ```

## Configuration

`config.yaml.example` 包含以下字段：

- `dataset_root`: 数据集根目录
- `dataset_yaml`: YOLO 数据配置文件路径
- `jpg_preview_dir`: 原始预览 JPG 文件夹路径
- `mask_dir`: 原始标签 PNG/掩码路径
- `model_path`: YOLO 模型权重文件路径
- `output_dir`: 训练输出目录
- `train_ratio`: 训练/验证划分比例
- `test_images_dir`: 推理测试图片目录

## Disclaimer

本项目代码已做脱敏处理，已移除个人服务器路径和隐私配置。使用前请自行创建并配置 `config.yaml`，确保所有路径指向本地合法目录。请勿将个人配置文件、模型权重或私有数据提交到公共仓库。

## Git 操作

```bash
git init
git add .
git commit -m "Initial import with sanitized config and docs"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```
