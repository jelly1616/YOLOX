from pathlib import Path
import yaml
import os


def load_final_config(path=None):
    if path is None:
        path = os.environ.get("YOLOX_CAN_CONFIG", "config_final.yaml")
    with Path(path).open('r', encoding='utf-8') as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError('config must be a YAML mapping')
    assert cfg['model']['classes'] == 1, 'This pipeline is single-class'
    assert len(cfg['model']['input_size']) == 2
    assert 0 < cfg['inference']['confidence'] < 1
    assert 0 < cfg['inference']['nms_iou'] < 1
    assert 0 < cfg['evaluation']['map_min_confidence'] < cfg['inference']['confidence']
    assert cfg['training']['no_aug_epochs'] < cfg['training']['epochs']
    return cfg
