# 기존 evaluate_can.py / predict_can.py 수정

## evaluate_can.py

상단 import에 다음을 추가:

```python
from config_loader import load_config
```

`main()`의 `parser = argparse.ArgumentParser()` 바로 앞에:

```python
cfg = load_config()
```

`parser.add_argument`의 기본값을 아래처럼 바꾸세요(기존 옵션 이름은 유지):

```python
parser.add_argument('--checkpoint', default=cfg['inference']['checkpoint'])
parser.add_argument('--annotations', default=f"{cfg['data']['directory']}/annotations/{cfg['data']['val_annotations']}")
parser.add_argument('--images', default=f"{cfg['data']['directory']}/{cfg['data']['val_images']}")
parser.add_argument('--confidence', type=float, default=cfg['inference']['confidence'])
parser.add_argument('--nms', type=float, default=cfg['inference']['nms_iou'])
parser.add_argument('--iou', type=float, default=cfg['evaluation']['matching_iou'])
parser.add_argument('--output', default=cfg['evaluation']['output_directory'])
```

`exp = Exp()` 다음에:

```python
exp.num_classes = cfg['model']['num_classes']
exp.input_size = tuple(cfg['model']['input_size'])
exp.test_size = tuple(cfg['model']['input_size'])
```

## predict_can.py

상단 import:

```python
from config_loader import load_config
cfg = load_config()
```

기존 상수를 다음으로 교체:

```python
MODEL_PATH = cfg['inference']['checkpoint']
IMAGE_DIR = Path(cfg['data']['directory']) / cfg['data']['val_images']
OUTPUT_DIR = Path(cfg['inference']['output_directory'])
CONF_THRESHOLD = cfg['inference']['confidence']
NMS_THRESHOLD = cfg['inference']['nms_iou']
MAX_IMAGES = cfg['inference']['max_images']
```

`exp = Exp()` 다음에:

```python
exp.num_classes = cfg['model']['num_classes']
exp.input_size = tuple(cfg['model']['input_size'])
exp.test_size = tuple(cfg['model']['input_size'])
```

검증은 `python evaluate_can.py` 와 `python predict_can.py`로 합니다.
