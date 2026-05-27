import os
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    yaml = None

DEFAULT_CONFIG = {
    "dataset_root": "./datasets",
    "dataset_yaml": "./dataset.yaml",
    "jpg_preview_dir": "./jpg_preview",
    "mask_dir": "./label",
    "model_path": "./yolov8n-seg.pt",
    "output_dir": "./runs",
    "train_ratio": 0.8,
    "test_images_dir": "./datasets/images/val",
}


def load_config(config_path: Optional[str] = None) -> dict:
    config_path = config_path or os.getenv("CONFIG_PATH", "config.yaml")
    config_file = Path(config_path)
    if not config_file.exists():
        return DEFAULT_CONFIG.copy()

    if yaml is None:
        raise ImportError(
            "PyYAML is required to load config.yaml. Install with `pip install pyyaml`."
        )

    with config_file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML mapping.")

    config = DEFAULT_CONFIG.copy()
    config.update({k: v for k, v in data.items() if v is not None})
    return config


if __name__ == "__main__":
    import json
    cfg = load_config()
    print(json.dumps(cfg, indent=2, ensure_ascii=False))
