"""Evaluate a fine-tuned single-class YOLOX-Nano on a COCO validation set.
Run from the YOLOX repository root.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from yolox.data.data_augment import ValTransform
from yolox.utils import postprocess

from can_nano import Exp


def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def match_boxes(gts, preds, iou_threshold):
    # Match high-scoring predictions first; one GT may match at most once.
    matched_gt = set()
    tp = 0
    for box, score in sorted(preds, key=lambda item: item[1], reverse=True):
        best_idx, best_iou = None, iou_threshold
        for j, gt in enumerate(gts):
            if j in matched_gt:
                continue
            iou = iou_xyxy(box, gt)
            if iou >= best_iou:
                best_iou, best_idx = iou, j
        if best_idx is not None:
            matched_gt.add(best_idx)
            tp += 1
    return tp, len(preds) - tp, len(gts) - tp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='training_outputs/can_nano_finetuned.pth')
    parser.add_argument('--annotations', default='datasets/CanCOCO/annotations/instances_val2017.json')
    parser.add_argument('--images', default='datasets/CanCOCO/val2017')
    parser.add_argument('--confidence', type=float, default=0.50)
    parser.add_argument('--nms', type=float, default=0.45)
    parser.add_argument('--iou', type=float, default=0.50)
    parser.add_argument('--output', default='evaluation_results')
    args = parser.parse_args()
    if not 0 < args.confidence < 1 or not 0 < args.nms < 1 or not 0 < args.iou <= 1:
        parser.error('confidence, nms and iou must be in (0,1] (confidence/nms < 1)')

    torch.set_num_threads(2)
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    coco = COCO(args.annotations)
    category_ids = coco.getCatIds()
    if len(category_ids) != 1:
        raise ValueError('Expected one can category')
    category_id = category_ids[0]
    exp = Exp()
    model = exp.get_model().cpu()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    preprocess = ValTransform(legacy=False)
    predictions_coco = []
    records = []
    total_tp = total_fp = total_fn = 0

    for image_id in sorted(coco.getImgIds()):
        entry = coco.loadImgs(image_id)[0]
        imgpath = Path(args.images) / entry['file_name']
        image = cv2.imread(str(imgpath))
        if image is None:
            raise FileNotFoundError(f'Cannot read: {imgpath}')
        h, w = image.shape[:2]
        ratio = min(exp.test_size[0] / h, exp.test_size[1] / w)
        processed, _ = preprocess(image, None, exp.test_size)
        tensor = torch.from_numpy(processed).unsqueeze(0).float()
        with torch.no_grad():
            output = model(tensor)
            detections = postprocess(output, 1, conf_thre=args.confidence, nms_thre=args.nms, class_agnostic=True)[0]

        pred_items = []
        if detections is not None:
            for det in detections.cpu().numpy():
                x1, y1, x2, y2 = (det[:4] / ratio).tolist()
                x1 = max(0.0, min(float(w), x1))
                y1 = max(0.0, min(float(h), y1))
                x2 = max(0.0, min(float(w), x2))
                y2 = max(0.0, min(float(h), y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                score = float(det[4] * det[5])
                box = (x1, y1, x2, y2)
                pred_items.append((box, score))
                predictions_coco.append({
                    'image_id': int(image_id), 'category_id': int(category_id),
                    'bbox': [x1, y1, x2-x1, y2-y1], 'score': score,
                })

        anns = coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))
        gt_boxes = []
        for ann in anns:
            if ann['category_id'] != category_id or ann.get('iscrowd', 0):
                continue
            x, y, bw, bh = ann['bbox']
            gt_boxes.append((x, y, x+bw, y+bh))
        tp, fp, fn = match_boxes(gt_boxes, pred_items, args.iou)
        total_tp += tp; total_fp += fp; total_fn += fn
        gt_count = len(gt_boxes); pred_count = len(pred_items)
        records.append({
            'file_name': entry['file_name'], 'gt_count': gt_count,
            'pred_count': pred_count, 'count_ratio_percent':
                round(100 * pred_count / gt_count, 2) if gt_count else 'N/A',
            'absolute_error': abs(pred_count - gt_count),
            'tp': tp, 'fp': fp, 'fn': fn,
        })
        print(f"{entry['file_name']}: GT={gt_count} Pred={pred_count} TP={tp} FP={fp} FN={fn}")

    with (outdir / 'count_per_image.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    (outdir / 'predictions_coco.json').write_text(json.dumps(predictions_coco, indent=2), encoding='utf-8')

    num = len(records)
    total_gt = sum(r['gt_count'] for r in records)
    total_pred = sum(r['pred_count'] for r in records)
    metrics = {
        'images': num,
        'confidence_threshold': args.confidence, 'nms_threshold': args.nms,
        'matching_iou_threshold': args.iou,
        'precision_at_threshold': total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0,
        'recall_at_threshold': total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0,
        'count_ratio_percent': 100 * total_pred / total_gt if total_gt else None,
        'count_mae': sum(r['absolute_error'] for r in records) / num,
        'exact_count_accuracy': sum(r['gt_count'] == r['pred_count'] for r in records) / num,
        'gt_total': total_gt, 'pred_total': total_pred,
    }
    if predictions_coco:
        dt = coco.loadRes(predictions_coco)
        evaluator = COCOeval(coco, dt, 'bbox')
        evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
        metrics['map_50_95'] = float(evaluator.stats[0])
        metrics['map_50'] = float(evaluator.stats[1])
        metrics['map_75'] = float(evaluator.stats[2])
    else:
        metrics['map_50_95'] = 0.0
        metrics['map_50'] = 0.0
        metrics['map_75'] = 0.0
    (outdir / 'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\n--- Evaluation Summary ---')
    for key, value in metrics.items():
        print(f'{key}: {value}')
    print(f'Files saved under: {outdir}')

if __name__ == '__main__':
    main()
