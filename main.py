"""One-command YOLOX-Nano dataset -> official training -> validation evaluation.

Run in YOLOX repository root. Designed to work with the project's existing
make_split_manifest.py, prepare_final_data.py, run_final_train.py, evaluate_final.py.

A chronological split is NOT a guarantee of independent validation: review the
manifest and explicitly approve it with --approve-split.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from final_config import load_final_config


REQUIRED_SCRIPTS = (
    "can_nano_final.py", "final_config.py", "make_split_manifest.py",
    "prepare_final_data.py", "run_final_train.py", "evaluate_final.py",
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def run_command(args: list[str], *, config_path: Path) -> None:
    env = os.environ.copy()
    env["YOLOX_CAN_CONFIG"] = str(config_path)
    print("\n$", " ".join(args), flush=True)
    subprocess.run(args, check=True, env=env)


def read_manifest(path: Path, images: list[Path], train_n: int, val_n: int) -> None:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"file_name", "group", "split"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Manifest must contain file_name,group,split")
        rows = list(reader)
    names = [row["file_name"] for row in rows]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate file names in split manifest")
    expected_names = {item.name for item in images}
    if set(names) != expected_names:
        missing, extra = expected_names - set(names), set(names) - expected_names
        raise ValueError(f"Manifest does not match images (missing={len(missing)}, extra={len(extra)})")
    count = {"train": 0, "val": 0}
    groups: dict[str, set[str]] = {}
    for row in rows:
        split, group = row["split"], row["group"]
        if split not in count or not group.strip():
            raise ValueError(f"Invalid manifest row: {row}")
        count[split] += 1
        groups.setdefault(group, set()).add(split)
    if count != {"train": train_n, "val": val_n}:
        raise ValueError(f"Expected train={train_n}, val={val_n}; got {count}")
    if any(len(splits) > 1 for splits in groups.values()):
        raise ValueError("A group appears in both train and validation")
    print(f"Manifest checked: {count['train']} train / {count['val']} val; {len(groups)} groups")
    print("IMPORTANT: this check cannot prove that adjacent video frames are independent.")


def validate_raw(cfg: dict) -> list[Path]:
    data = cfg["data"]
    folder = Path(data["raw_images"])
    labels = Path(data["raw_labels"])
    if not folder.is_dir() or not labels.is_dir():
        raise FileNotFoundError(f"Required raw image/label folders missing: {folder}, {labels}")
    images = sorted(x for x in folder.iterdir() if x.is_file() and x.suffix.lower() in IMAGE_EXTENSIONS)
    total = data["train_count"] + data["val_count"]
    if len(images) != total:
        raise ValueError(f"Expected {total} images, found {len(images)} in {folder}")
    basenames = [p.stem for p in images]
    if len(basenames) != len(set(basenames)):
        raise ValueError("Two images have the same stem (e.g. a.jpg and a.png)")
    missing = [p.name for p in images if not (labels / (p.stem + ".txt")).is_file()]
    if missing:
        raise ValueError(f"Missing {len(missing)} YOLO TXT labels; first examples: {missing[:5]}")
    print(f"Raw input OK: {len(images)} images and matching labels")
    return images


def check_converted(cfg: dict) -> bool:
    root = Path(cfg["data"]["root"])
    if not root.is_dir():
        return False
    for subset, expected in (("train", cfg["data"]["train_count"]), ("val", cfg["data"]["val_count"])):
        annotation = root / "annotations" / f"instances_{subset}2017.json"
        if not annotation.is_file():
            raise RuntimeError(f"Incomplete existing dataset: missing {annotation}")
        contents = json.loads(annotation.read_text(encoding="utf-8"))
        if len(contents.get("images", [])) != expected:
            raise ValueError(f"Existing {annotation} has wrong image count; do not reuse")
        image_dir = root / f"{subset}2017"
        if any(not (image_dir / item["file_name"]).is_file() for item in contents["images"]):
            raise ValueError(f"Missing converted images under {image_dir}")
    existing_manifest = root / "split_manifest.csv"
    source_manifest = Path(cfg["data"]["manifest"])
    if not existing_manifest.is_file() or existing_manifest.read_bytes() != source_manifest.read_bytes():
        raise ValueError("Converted dataset does not match manifest; do not reuse it")
    print("Existing converted COCO dataset verified; conversion skipped.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOX-Nano 5000-image end-to-end pipeline")
    parser.add_argument("--config", default="config_final.yaml", help="YAML configuration file")
    parser.add_argument("--approve-split", action="store_true", help="I reviewed the recording/time-based train/val split")
    parser.add_argument("--group-size", type=int, default=100, help="Draft chronological grouping size")
    parser.add_argument("--prepare-only", action="store_true", help="Only validate/convert the data")
    parser.add_argument("--check-only", action="store_true", help="Only verify raw data and manifest")
    parser.add_argument("--resume", action="store_true", help="Resume a saved official Trainer checkpoint")
    parser.add_argument("--evaluate-only", action="store_true", help="Run evaluation on an existing trained checkpoint")
    parser.add_argument("--checkpoint", help="Explicit checkpoint for --evaluate-only (or to override automatic selection)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help="Evaluation device")
    args = parser.parse_args()

    if args.resume and (args.prepare_only or args.evaluate_only or args.check_only):
        parser.error("--resume cannot be used with --prepare-only, --evaluate-only or --check-only")
    if args.checkpoint and not args.evaluate_only:
        parser.error("--checkpoint is for --evaluate-only")
    root = Path(__file__).resolve().parent
    os.chdir(root)
    for name in REQUIRED_SCRIPTS:
        if not (root / name).is_file():
            raise FileNotFoundError(f"Place {name} next to main.py")

    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    cfg = load_final_config(str(config_path))
    exp_name = cfg["project"]["experiment"]
    print(f"Experiment: {exp_name}; config: {config_path}", flush=True)
    print(f"Data: train={cfg['data']['train_count']}, val={cfg['data']['val_count']}")
    # Validate that the Exp used by tools/train.py reads this exact YAML.
    env = os.environ.copy()
    env["YOLOX_CAN_CONFIG"] = str(config_path)
    exp_probe = (
        "from can_nano_final import Exp; e=Exp(); "
        "print('EXP_CONFIG:', e.exp_name, e.data_dir, e.max_epoch, e.input_size); "
        f"assert e.exp_name == {exp_name!r}; "
        f"assert str(e.data_dir) == {str(cfg['data']['root'])!r}"
    )
    subprocess.run([sys.executable, "-c", exp_probe], check=True, env=env)
    if not (args.prepare_only or args.check_only or args.evaluate_only):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: no training started. Use --prepare-only for CPU data preparation.")

    if not args.evaluate_only:
        images = validate_raw(cfg)
        manifest = Path(cfg["data"]["manifest"])
        if not manifest.is_file():
            if args.resume:
                raise FileNotFoundError("Split manifest missing during resume")
            if args.group_size <= 0:
                raise ValueError("group-size must be positive")
            run_command([sys.executable, "make_split_manifest.py", "--config", str(config_path), "--group-size", str(args.group_size)], config_path=config_path)
            print(f"\nGenerated split DRAFT: {manifest}")
            if not args.approve_split:
                print("STOP: inspect file_name/group/split, ensure train and val do not share near-identical frames.")
                print("Then execute again with --approve-split (no training has started).")
                return
        read_manifest(manifest, images, cfg["data"]["train_count"], cfg["data"]["val_count"])
        if args.check_only:
            print("Raw data / manifest checks passed. No conversion or training started.")
            return
        if not args.approve_split:
            raise RuntimeError("Explicit --approve-split is required to use continuous-video frames")
        if not check_converted(cfg):
            run_command([sys.executable, "prepare_final_data.py", "--config", str(config_path), "--dry-run"], config_path=config_path)
            run_command([sys.executable, "prepare_final_data.py", "--config", str(config_path)], config_path=config_path)
        run_command([sys.executable, "run_final_train.py", "--config", str(config_path), "--check"], config_path=config_path)
        if args.prepare_only:
            print("Preparation completed. No training started.")
            return
        cmd = [sys.executable, "run_final_train.py", "--config", str(config_path)]
        if args.resume:
            cmd.append("--resume")
        run_command(cmd, config_path=config_path)

    output_dir = Path(cfg["output"]["root"]) / exp_name
    if args.checkpoint:
        chosen = Path(args.checkpoint)
    else:
        chosen = next((p for p in (output_dir / "best_ckpt.pth", output_dir / "latest_ckpt.pth") if p.is_file()), None)
    if chosen is None or not chosen.is_file():
        raise FileNotFoundError(f"No trained checkpoint found in {output_dir}. Evaluation did not start.")
    report_dir = output_dir / "evaluation_final"
    if report_dir.exists() and any(report_dir.iterdir()):
        raise FileExistsError(f"Evaluation directory already contains results: {report_dir}. Move it aside first.")
    run_command([
        sys.executable, "evaluate_final.py", "--config", str(config_path),
        "--checkpoint", str(chosen), "--device", args.device,
        "--output", str(report_dir),
    ], config_path=config_path)
    print(f"\nSUCCESS: evaluation saved under {report_dir}")
    print("Reminder: these are VALIDATION metrics, not independent test performance.")


if __name__ == "__main__":
    main()
