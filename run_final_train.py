
"""
YOLOX-Nano 최종 학습 실행 프로그램

기능:
1. YAML 설정 파일 선택
2. Train / Validation 데이터셋 사전 검증
3. CUDA GPU 환경 확인
4. 사전학습 가중치 파일 검사
5. 기존 학습 결과 덮어쓰기 방지
6. 학습 설정 YAML 백업
7. 공식 YOLOX Trainer 실행
8. 체크포인트를 이용한 학습 재개

실행 예시:

# 테스트 데이터셋 사전 검증
python run_final_train.py --config config_test.yaml --check

# 최종 데이터셋 사전 검증
python run_final_train.py --config config_final.yaml --check

# GPU 컴퓨터에서 새 학습 시작
python run_final_train.py --config config_final.yaml

# 기존 학습 이어서 진행
python run_final_train.py --config config_final.yaml --resume
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

from pathlib import Path

import torch

from final_config import load_final_config


def validate_dataset(cfg):
    """
    Train / Validation 데이터셋 검증
    """

    root = Path(cfg["data"]["root"])
    ann_dir = root / "annotations"

    datasets = [
        ("train", cfg["data"]["train_count"]),
        ("val", cfg["data"]["val_count"]),
    ]

    for name, expected in datasets:

        # COCO JSON 경로
        ann_path = ann_dir / f"instances_{name}2017.json"

        if not ann_path.is_file():
            raise FileNotFoundError(
                f"Annotation file not found: {ann_path}"
            )

        # COCO JSON 읽기
        obj = json.loads(
            ann_path.read_text(encoding="utf-8")
        )

        images = obj["images"]
        annotations = obj["annotations"]

        # 이미지 수 확인
        if len(images) != expected:
            raise ValueError(
                f"{ann_path}: "
                f"expected {expected} images, "
                f"got {len(images)}"
            )

        # 실제 이미지 파일 확인
        image_dir = root / f"{name}2017"

        image_ids = set()

        for item in images:
            image_path = image_dir / item["file_name"]

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )

            image_ids.add(item["id"])

        # 이미지별 캔 라벨 수 계산
        annotation_counts = {}

        for item in annotations:
            image_id = item["image_id"]

            if image_id not in image_ids:
                raise ValueError(
                    f"Annotation references unknown image: {image_id}"
                )

            annotation_counts[image_id] = (
                annotation_counts.get(image_id, 0) + 1
            )

        max_labels = max(
            annotation_counts.values(),
            default=0,
        )

        print(
            f"{name}: "
            f"{len(images)} images / "
            f"{len(annotations)} cans / "
            f"max cans per frame={max_labels}"
        )

        # 최대 라벨 수 확인
        configured_max = cfg["training"]["max_labels"]

        if max_labels > configured_max:
            raise ValueError(
                f"{name}: maximum labels {max_labels} "
                f"exceeds configured max_labels={configured_max}"
            )

    print("Dataset validation completed.")


def validate_gpu(cfg):
    """
    CUDA GPU 환경 검증
    """

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Official YOLOX Trainer requires CUDA. "
            "Use the GPU training computer. "
            "On the laptop, use --check."
        )

    requested_gpus = cfg["training"]["gpus"]
    available_gpus = torch.cuda.device_count()

    print("Available GPUs:", available_gpus)
    print("Requested GPUs:", requested_gpus)

    if requested_gpus < 1:
        raise ValueError("GPU count must be at least 1")

    if available_gpus < requested_gpus:
        raise RuntimeError(
            f"Requested {requested_gpus} GPUs, "
            f"but only {available_gpus} are available."
        )

    for index in range(requested_gpus):
        print(
            f"GPU {index}:",
            torch.cuda.get_device_name(index),
        )


def main():

    # --------------------------------------------------
    # 1. 실행 인자
    # --------------------------------------------------

    parser = argparse.ArgumentParser(
        description="YOLOX-Nano final training pipeline"
    )

    parser.add_argument(
        "--config",
        default="config_final.yaml",
        help="YAML configuration file",
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate settings and dataset without training",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume saved official YOLOX checkpoint",
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # 2. YAML 설정 로드
    # --------------------------------------------------

    config_path = Path(args.config).resolve()

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    cfg = load_final_config(str(config_path))

    experiment_name = cfg["project"]["experiment"]

    print("=" * 60)
    print("YOLOX-Nano Training Configuration")
    print("=" * 60)

    print("Experiment:", experiment_name)
    print("Config:", config_path)
    print("Input size:", cfg["model"]["input_size"])
    print("Epochs:", cfg["training"]["epochs"])
    print("Batch size:", cfg["training"]["batch_size"])
    print("Base learning rate:", cfg["training"]["base_learning_rate"])
    print("Mosaic probability:", cfg["training"]["mosaic_probability"])
    print("MixUp enabled:", cfg["training"]["mixup_enabled"])

    # --------------------------------------------------
    # 3. 데이터셋 검증
    # --------------------------------------------------

    validate_dataset(cfg)

    # --------------------------------------------------
    # 4. Preflight 모드
    # --------------------------------------------------

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if args.check:
        print()
        print("Preflight OK. Training has NOT started.")
        return

    # --------------------------------------------------
    # 5. CUDA GPU 검증
    # --------------------------------------------------

    validate_gpu(cfg)

    # --------------------------------------------------
    # 6. 실험 결과 저장 경로
    # --------------------------------------------------

    output_dir = (
        Path(cfg["output"]["root"])
        / experiment_name
    )

    # --------------------------------------------------
    # 7. 사전학습 가중치 / 재개 체크포인트 검증
    # --------------------------------------------------

    pretrained = None

    if args.resume:

        # 학습 재개 모드
        checkpoint_path = output_dir / "latest_ckpt.pth"

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {checkpoint_path}"
            )

        print("Resume checkpoint:", checkpoint_path)

    else:

        # 새로운 학습 모드
        pretrained = Path(cfg["model"]["pretrained"])

        if not pretrained.is_file():
            raise FileNotFoundError(
                f"Pretrained checkpoint not found: {pretrained}"
            )

        # 기존 실험 결과 덮어쓰기 방지
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"Experiment directory already contains files: "
                f"{output_dir}"
            )

        print("Pretrained checkpoint:", pretrained)

    # --------------------------------------------------
    # 8. YAML 설정 백업 및 동일성 확인
    # --------------------------------------------------

    saved_config = output_dir / "config_used.yaml"

    if args.resume:

        if not saved_config.is_file():
            raise FileNotFoundError(
                f"Saved configuration not found: {saved_config}"
            )

        # 이전 학습과 현재 YAML 설정 비교
        if config_path.read_bytes() != saved_config.read_bytes():
            raise ValueError(
                "Current YAML configuration differs from "
                "the saved training configuration. "
                "Verify settings before resuming."
            )

        print("Saved configuration verified.")

    else:

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            config_path,
            saved_config,
        )

        print("Configuration backup saved:", saved_config)

    # --------------------------------------------------
    # 9. 공식 YOLOX Trainer 실행 명령어
    # --------------------------------------------------

    gpus = cfg["training"]["gpus"]
    batch_size = cfg["training"]["batch_size"]

    command = [
        sys.executable,
        "tools/train.py",
        "-f",
        "can_nano_final.py",
        "-d",
        str(gpus),
        "-b",
        str(batch_size),
        "--logger",
        "tensorboard",
    ]

    # FP16 혼합 정밀도
    if cfg["training"]["fp16"]:
        command.append("--fp16")

    # 학습 재개 또는 사전학습 가중치 적용
    if args.resume:
        command.append("--resume")
    else:
        command.extend([
            "-c",
            str(pretrained),
        ])

    # --------------------------------------------------
    # 10. 동일 YAML을 공식 Trainer에 전달
    # --------------------------------------------------

    env = os.environ.copy()

    env["YOLOX_CAN_CONFIG"] = str(config_path)

    print()
    print("=" * 60)
    print("Starting Official YOLOX Trainer")
    print("=" * 60)

    print("Command:", " ".join(command))
    print("Experiment:", experiment_name)
    print("Output:", output_dir)

    # --------------------------------------------------
    # 11. 실제 학습 실행
    # --------------------------------------------------

    subprocess.run(
        command,
        check=True,
        env=env,
    )

    print()
    print("Training completed successfully.")
    print("Results saved under:", output_dir)


if __name__ == "__main__":
    main()
