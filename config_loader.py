"""Load and validate YOLOX-Nano can project YAML settings."""
from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)
    if not isinstance(config, dict):
        raise ValueError("config.yaml must contain YAML mappings")
    for key in ("model", "data", "training", "inference", "evaluation"):
        if not isinstance(config.get(key), dict):
            raise ValueError(f"Missing config section: {key}")
    size = config["model"]["input_size"]
    if not isinstance(size, list) or len(size) != 2 or not all(isinstance(x, int) and x > 0 and x % 32 == 0 for x in size):
        raise ValueError("model.input_size must be two positive multiples of 32")
    for key, val in (("inference.confidence", config["inference"]["confidence"]),
                     ("inference.nms_iou", config["inference"]["nms_iou"]),
                     ("evaluation.matching_iou", config["evaluation"]["matching_iou"])):
        if not isinstance(val, (float, int)) or not 0 < val <= 1:
            raise ValueError(f"{key} must be in (0, 1]")
    return config
