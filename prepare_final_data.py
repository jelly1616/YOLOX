"""Convert YOLOLabel -> COCO with REQUIRED video-group split manifest.
CSV columns: file_name,group,split, e.g. 000001.jpg,session_A,train
All members of one group MUST have same split. Exactly 4000 train and 1000 val.
One video can be split into temporally separated groups, but nearby-frame leakage
is still possible: use held-out recording sessions whenever possible.
"""
import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image
from final_config import load_final_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config_final.yaml')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    cfg = load_final_config(args.config)
    d = cfg['data']
    image_dir, label_dir = Path(d['raw_images']), Path(d['raw_labels'])
    manifest = Path(d['manifest'])
    out = Path(d['root'])
    if out.exists() and not args.dry_run:
        raise FileExistsError(f'{out} exists: move it aside before generating a fresh dataset')
    with manifest.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not {'file_name', 'group', 'split'} <= set(reader.fieldnames or []):
            raise ValueError('Manifest columns required: file_name,group,split')
        rows = list(reader)
    if len(rows) != len({r['file_name'] for r in rows}):
        raise ValueError('Duplicate image name in manifest')
    groups = defaultdict(set)
    counts = Counter()
    for row in rows:
        if row['split'] not in ('train', 'val') or not row['group']:
            raise ValueError(f'Invalid manifest record: {row}')
        groups[row['group']].add(row['split'])
        counts[row['split']] += 1
    if any(len(v) != 1 for v in groups.values()):
        raise ValueError('Video group overlaps train and val: group leakage')
    if counts != Counter({'train': d['train_count'], 'val': d['val_count']}):
        raise ValueError(f'Expected exact 4000/1000 (from YAML), got {counts}')
    if len(groups) < 2:
        raise ValueError('At least two separate recording groups required')
    planned = {'train': {'images': [], 'annotations': []}, 'val': {'images': [], 'annotations': []}}
    ann_id = 1
    max_boxes = 0
    for image_id, row in enumerate(rows, start=1):
        filename = row['file_name']
        if '/' in filename or '\\' in filename or Path(filename).name != filename:
            raise ValueError('Manifest file_name must be a flat filename')
        img = image_dir / filename
        label = label_dir / (img.stem + '.txt')
        if not img.is_file() or not label.is_file():
            raise FileNotFoundError(f'Missing image or label: {img} / {label}')
        with Image.open(img) as im:
            width, height = im.size
            im.verify()
        subset = planned[row['split']]
        subset['images'].append({'id': image_id, 'file_name': filename, 'width': width, 'height': height})
        num = 0
        for line_num, line in enumerate(label.read_text(encoding='utf-8-sig').splitlines(), start=1):
            if not line.strip():
                continue
            fields = line.split()
            try:
                if len(fields) != 5:
                    raise ValueError('expected 5 fields')
                cls, xc, yc, bw, bh = map(float, fields)
                if cls != 0 or not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                    raise ValueError('invalid class or normalized box')
                x1, y1 = max(0, (xc-bw/2)*width), max(0, (yc-bh/2)*height)
                x2, y2 = min(width, (xc+bw/2)*width), min(height, (yc+bh/2)*height)
                if x2 <= x1 or y2 <= y1:
                    raise ValueError('box outside image')
            except ValueError as exc:
                raise ValueError(f'{label}:{line_num}: {exc}') from exc
            w, h = x2-x1, y2-y1
            subset['annotations'].append({'id': ann_id, 'image_id': image_id, 'category_id': 1, 'bbox': [x1, y1, w, h], 'area': w*h, 'iscrowd': 0, 'segmentation': []})
            ann_id += 1
            num += 1
        max_boxes = max(max_boxes, num)
    if max_boxes * 4 > cfg['training']['max_labels'] and cfg['training']['mosaic_probability'] > 0:
        print('WARNING: 4x max per-frame GT exceeds max_labels; mosaic may truncate labels. Raise max_labels or turn mosaic off.')
    print('Counts:', dict(counts), '| groups:', len(groups), '| maximum cans/frame:', max_boxes)
    if args.dry_run:
        print('Dry-run finished; no files written.')
        return
    (out / 'annotations').mkdir(parents=True)
    for split, obj in planned.items():
        (out / f'{split}2017').mkdir()
        for entry in obj['images']:
            shutil.copy2(image_dir / entry['file_name'], out / f'{split}2017' / entry['file_name'])
        output = {**obj, 'info': {'description': 'can detection / video group split'}, 'licenses': [], 'categories': [{'id': 1, 'name': 'can', 'supercategory': 'object'}]}
        (out / 'annotations' / f'instances_{split}2017.json').write_text(json.dumps(output, ensure_ascii=False), encoding='utf-8')
    shutil.copy2(manifest, out / 'split_manifest.csv')
    print('Finished:', out)


if __name__ == '__main__':
    main()
