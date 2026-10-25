"""Convert YOLOLabel TXT annotations to YOLOX-compatible COCO datasets.

Run in the YOLOX repository root:
  python convert_can_dataset.py
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from PIL import Image

RAW = Path('datasets/can_raw')
OUT = Path('datasets/CanCOCO')
SEED = 42
SPLITS = {'train': 0.70, 'val': 0.15, 'test': 0.15}
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

def convert():
    image_dir = RAW / 'images'
    label_dir = RAW / 'labels'
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f'이미지 및 라벨 폴더를 확인해 주세요: {image_dir}, {label_dir}')

    images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not images:
        raise ValueError(f'이미지가 없습니다: {image_dir}')

    stems = [p.stem for p in images]
    if len(set(stems)) != len(stems):
        raise ValueError('확장자만 다른 동일 파일명이 있습니다. 이미지 이름을 고유하게 변경해 주세요.')

    missing = [p.name for p in images if not (label_dir / (p.stem + '.txt')).exists()]
    if missing:
        raise FileNotFoundError(f'대응되는 라벨 TXT가 없습니다: {missing[:10]} (총 {len(missing)}개)')

    random.Random(SEED).shuffle(images)
    n = len(images)
    n_train = int(n * SPLITS['train'])
    n_val = int(n * SPLITS['val'])
    subsets = {
        'train': images[:n_train],
        'val': images[n_train:n_train + n_val],
        'test': images[n_train + n_val:],
    }

    # Read and validate everything before writing outputs.
    prepared = {}
    next_ann_id = 1
    for split, files in subsets.items():
        coco_images, coco_annotations, copy_jobs = [], [], []
        for img_id, image_path in enumerate(files, start=1):
            with Image.open(image_path) as im:
                width, height = im.size
                im.verify()
            label_path = label_dir / (image_path.stem + '.txt')
            coco_images.append({'id': img_id, 'file_name': image_path.name, 'width': width, 'height': height})
            for lineno, line in enumerate(label_path.read_text(encoding='utf-8-sig').splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                try:
                    if len(parts) != 5:
                        raise ValueError('5개 값이 필요합니다')
                    cls, xc, yc, bw, bh = map(float, parts)
                    if cls != 0 or not all(0 <= v <= 1 for v in (xc, yc, bw, bh)) or bw <= 0 or bh <= 0:
                        raise ValueError('클래스는 0이어야 하며, 정규화 좌표는 0~1이고 너비/높이는 양수여야 합니다')
                    x1 = max(0.0, (xc - bw / 2) * width)
                    y1 = max(0.0, (yc - bh / 2) * height)
                    x2 = min(float(width), (xc + bw / 2) * width)
                    y2 = min(float(height), (yc + bh / 2) * height)
                    if x2 <= x1 or y2 <= y1:
                        raise ValueError('이미지 안에 유효한 박스가 없습니다')
                except ValueError as exc:
                    raise ValueError(f'{label_path}:{lineno} 라벨 오류: {exc}') from exc
                box_w, box_h = x2 - x1, y2 - y1
                coco_annotations.append({
                    'id': next_ann_id,
                    'image_id': img_id,
                    'category_id': 1,
                    'bbox': [round(x1, 4), round(y1, 4), round(box_w, 4), round(box_h, 4)],
                    'area': round(box_w * box_h, 4),
                    'iscrowd': 0,
                    'segmentation': [],
                })
                next_ann_id += 1
            copy_jobs.append((image_path, OUT / f'{split}2017' / image_path.name))
        prepared[split] = ({
            'info': {'description': 'Can detection dataset (YOLOLabel converted to COCO)'},
            'licenses': [],
            'images': coco_images,
            'annotations': coco_annotations,
            'categories': [{'id': 1, 'name': 'can', 'supercategory': 'object'}],
        }, copy_jobs)

    if OUT.exists():
        raise FileExistsError(f'{OUT} 폴더가 이미 있습니다. 기존 변환 결과를 확인하거나 이름을 바꾼 뒤 재실행하세요.')
    (OUT / 'annotations').mkdir(parents=True)
    for split, (coco, jobs) in prepared.items():
        (OUT / f'{split}2017').mkdir()
        for src, dst in jobs:
            shutil.copy2(src, dst)
        annotation_file = OUT / 'annotations' / f'instances_{split}2017.json'
        annotation_file.write_text(json.dumps(coco, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'{split:5} | 이미지 {len(coco["images"]):3}장 | 캔 라벨 {len(coco["annotations"]):5}개 | {annotation_file}')
    print('\n완료! 원본 images/labels 파일은 변경되지 않았습니다.')
    print('주의: 동일 촬영 영상의 연속 프레임이 섞여 있다면 최종 평가는 영상 단위로 다시 분할해야 합니다.')


if __name__ == '__main__':
    convert()
