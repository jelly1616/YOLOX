# YOLOX-Nano 5,000장 공식 학습 파이프라인 (개발 초안)

> **상태:** Python 문법 검사 완료. 사용자의 실제 YOLOX 리비전·Windows 노트북·NVIDIA GPU PC에서는 아직 통합 실행 검증되지 않았습니다. YOLOX 공식 `Trainer`는 CUDA 전용이므로 GPU 학습 컴퓨터에서 테스트해야 합니다.

## 사용 조건

- Train **4,000장**, Validation **1,000장**. Test 없음.
- 원본 1600×1200 이미지는 그대로 저장; 모델 입력은 **비율 유지 축소 + 패딩 416×416**.
- 단일 클래스 `can`, 원래 YOLOLabel class ID `0` → COCO category ID `1`.
- 반드시 출력: **AP50, mAP@50:95, Precision@Confidence0.5, Recall@Confidence0.5, Can Count Ratio@Confidence0.5**. 추가로 Count MAE와 Exact Count Accuracy.
- 연속 촬영 영상에서 생성한 프레임의 경우 **무작위 이미지 분할 금지**. `split_manifest.csv`의 그룹/시간 구간을 직접 검토.

## 넣을 위치

이 ZIP의 `.py` 파일과 `config_final.yaml`을 기존 `can_detection/YOLOX/` 루트에 놓으세요. 기존 `train_can.py`, `evaluate_can.py`, `config.yaml`은 덮어쓰지 않습니다.

```
YOLOX/
  can_nano_final.py
  config_final.yaml
  final_config.py
  prepare_final_data.py
  make_split_manifest.py
  run_final_train.py
  evaluate_final.py
  datasets/can_raw/images/*.jpg
  datasets/can_raw/labels/*.txt
  weights/yolox_nano.pth
```

## 1. 파일 확인

Git Bash, YOLOX 루트에서:

```bash
python -m pip install PyYAML
python -m compileall -q can_nano_final.py final_config.py prepare_final_data.py make_split_manifest.py run_final_train.py evaluate_final.py
python -c "from can_nano_final import Exp; e=Exp(); print(e.num_classes, e.input_size, e.max_epoch, e.basic_lr_per_img)"
```

## 2. GPU 학습 컴퓨터 사양 확인

GPU PC에서 `nvidia-smi` 및 아래 실행:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

GPU에서 사용할 PyTorch/CUDA 호환 버전과 공식 YOLOX 의존성은 해당 PC 기준으로 구성하세요. 현재 Windows CPU 노트북의 PyTorch 버전을 그대로 복제할 필요는 없습니다. `fp16`은 GPU 검증 전에는 false 유지.

## 3. Train/Val 분할 (중요)

5,000장의 원본을 `datasets/can_raw/images` 및 `datasets/can_raw/labels`에 배치합니다. 원본 파일을 수정하지 않습니다.

**권장:** 촬영 구간/세션 기준으로 별도의 4,000 Train / 1,000 Val 이미지를 지정한 `datasets/split_manifest.csv`를 직접 생성:

```csv
file_name,group,split
000001.jpg,recording_20261020_AM,train
000002.jpg,recording_20261020_AM,train
000503.jpg,recording_20261021_PM,val
```

- 한 `group` 값이 Train과 Val에 동시에 등장하면 변환 스크립트가 오류로 중단합니다.
- 5,000장 모두 한 영상에서 연속 캡처한 경우 완전한 독립 검증으로 보기는 어렵습니다. 가능하면 **다른 촬영 세션의 1,000장**을 Validation에 배정하세요.
- 원본 이름이 시간순으로 정렬되는 것이 **확실한 경우에만** 임시 초안 생성 가능:

```bash
python make_split_manifest.py
```

이 명령은 정렬된 앞 4,000장을 Train, 뒤 1,000장을 Val로 나눕니다. 인접 프레임 유출 및 한 세션 편향 가능성이 있으므로 파일을 검토하고 필요하면 수정해야 합니다. 기존 manifest는 덮어쓰지 않습니다.

검증 후 데이터 변환:

```bash
python prepare_final_data.py --dry-run
python prepare_final_data.py
```

결과 폴더 `datasets/CanCOCO_final`이 이미 있다면 안전을 위해 덮어쓰지 않고 중단합니다. 기존 200장 `datasets/CanCOCO`와 별도 보관됩니다.

## 4. YAML 확정

`config_final.yaml`에서 팀원과 합의한 **epochs, batch_size, base_learning_rate, weight_decay, augmentation**을 입력하세요. `training.base_learning_rate`는 해당 배치 크기에 대한 **실제 기준 LR**로 지정하고, 내부에서 `basic_lr_per_img = LR / batch_size`로 변환합니다. 나중에 배치를 바꾸면 LR 동작을 함께 검토해야 합니다.

원본 영상에 객체가 많아 공식 로더의 50/120개 제한을 그대로 쓰지 않았습니다. `max_labels=300`은 임시값이며, 4장 Mosaic 결합으로 초과할 수 있으므로 실제 라벨 수 분포를 보고 충분히 늘리거나 Mosaic을 0으로 끄세요.

## 5. 학습 컴퓨터에서 사전 점검 후 공식 학습

```bash
python run_final_train.py --check
python run_final_train.py
```

내부 실행은 `tools/train.py -f can_nano_final.py ... -c weights/yolox_nano.pth`입니다. **공식 YOLOX Trainer는 CUDA를 요구**하기 때문에 개인 CPU 노트북에서는 `--check`까지만 테스트합니다.

실행 중단 뒤 (체크포인트가 생성된 경우):

```bash
python run_final_train.py --resume
```

기본 저장 경로는 `YOLOX_outputs/can_nano_5000_v1/`이며, `train_log.txt`, `tensorboard/`, `latest_ckpt.pth`, `best_ckpt.pth` 등이 생성됩니다. 가장 좋은 모델은 **Validation mAP@50:95 기준**으로 선택합니다. 설정에 따라 파일 종류는 달라질 수 있습니다.

## 6. 최종 평가 (Validation 1,000장)

```bash
python evaluate_final.py --checkpoint YOLOX_outputs/can_nano_5000_v1/best_ckpt.pth --device cuda
```

GPU가 없는 PC로 체크포인트를 옮겨서 평가하는 것도 가능합니다 (`--device cpu`). `final_evaluation/` 아래 `metrics.json`, `metrics.csv`, `count_per_image.csv`, `predictions_coco.json` 생성.

평가 정의:

- **AP50, mAP@50:95**: COCOeval, 낮은 `map_min_confidence`(기본 0.001)에서 계산. 일반적인 AP와 동일한 **검출 수 상한(maxDets)** 조건을 함께 보고합니다.
- **Precision, Recall**: Confidence 0.5, 예측-정답 1:1 IoU 0.5 매칭 기준.
- **Can Count Ratio**: `100 × (Confidence 0.5를 통과한 예측 박스 총수) / (Validation 정답 박스 총수)`; 100%를 넘을 수 있으며 순수한 정확도와 다릅니다.
- **Count MAE**: 이미지당 `abs(pred_count - gt_count)` 평균.
- Validation을 사용해 모델 선택과 하이퍼파라미터 변경을 했으므로 최종 독립 Test 정확도로 부르면 안 됩니다. 추후 별도 촬영 영상으로 독립 검증을 권장합니다.

## 7. Git

사용자는 이미 `feature/final-training-pipeline` 브랜치에 있습니다. 먼저 파일을 검토한 뒤 코드만 추가:

```bash
git status -sb
git add config_final.yaml final_config.py can_nano_final.py prepare_final_data.py make_split_manifest.py run_final_train.py evaluate_final.py README_START_HERE.md
git diff --cached --stat
git commit -m "feat: add final YOLOX-Nano training and validation pipeline"
git push -u origin feature/final-training-pipeline
```

`datasets/`, `YOLOX_outputs/`, `weights/`는 기존 `.gitignore`에서 제외됩니다. 개인 데이터나 모델을 `git add -f`로 강제 추가하지 마세요.

## 아직 확인해야 할 점

- GPU 하드웨어/드라이버/CUDA/PyTorch 호환성.
- 설치된 YOLOX 커밋에서 공식 Trainer, DataLoader API 호환성 및 첫 Epoch 실행.
- 연속 프레임에 대한 그룹별 분할 품질, 4,000/1,000 구성의 독립성.
- `max_labels`, Mosaic 등으로 라벨이 잘리지 않는지 확인.
- 코드 통합 이후 1 Epoch 스모크 테스트 / Checkpoint 재개 테스트.
- 최종 Raspberry Pi 5 실시간 10 FPS 및 누적 캔 계수는 이 파이프라인에 포함하지 않습니다.
