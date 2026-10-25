"""Independent validation: AP50/mAP from low-cutoff detections;
Precision, Recall and Can Count Ratio at operating confidence 0.50.
Output is validation-set performance, not a separate held-out final test.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from yolox.data.data_augment import ValTransform
from yolox.utils import postprocess
from can_nano_final import Exp
from final_config import load_final_config


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1)*max(0, y2-y1)
    ar_a = max(0, a[2]-a[0])*max(0, a[3]-a[1])
    ar_b = max(0, b[2]-b[0])*max(0, b[3]-b[1])
    return inter/(ar_a+ar_b-inter) if ar_a+ar_b-inter > 0 else 0.0


def match(gts, preds, threshold):
    matched = set()
    tp = 0
    for box, score in sorted(preds, key=lambda p: p[1], reverse=True):
        candidates = [(iou(box, gt), j) for j, gt in enumerate(gts) if j not in matched]
        if candidates:
            best, j = max(candidates)
            if best >= threshold:
                matched.add(j)
                tp += 1
    return tp, len(preds)-tp, len(gts)-tp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', default='config_final.yaml')
    parser.add_argument('--output', default='final_evaluation')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    cfg = load_final_config(args.config)
    torch.set_num_threads(2)
    dev = torch.device(args.device)
    if dev.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    root = Path(cfg['data']['root'])
    coco = COCO(str(root / 'annotations' / 'instances_val2017.json'))
    category_ids = coco.getCatIds()
    if len(category_ids) != 1:
        raise ValueError('Expected one can class')
    exp = Exp()
    model = exp.get_model().to(dev)
    checkpoint = torch.load(args.checkpoint, map_location=dev, weights_only=True)
    # Official YOLOX checkpoints generally put the model (EMA during training) under 'model'.
    model.load_state_dict(checkpoint['model'])
    model.eval()
    transform = ValTransform(legacy=False)
    conf = cfg['inference']['confidence']
    low_conf = cfg['evaluation']['map_min_confidence']
    match_iou = cfg['evaluation']['matching_iou']
    max_dets = cfg['evaluation']['max_detections']
    predictions = []
    records = []
    total_tp = total_fp = total_fn = 0
    for image_id in sorted(coco.getImgIds()):
        entry = coco.loadImgs(image_id)[0]
        path = root / 'val2017' / entry['file_name']
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        h, w = image.shape[:2]
        ratio = min(exp.test_size[0]/h, exp.test_size[1]/w)
        processed, _ = transform(image, None, exp.test_size)
        x = torch.from_numpy(processed).unsqueeze(0).float().to(dev)
        with torch.no_grad():
            boxes = postprocess(model(x), 1, conf_thre=low_conf, nms_thre=cfg['inference']['nms_iou'], class_agnostic=True)[0]
        operating = []
        if boxes is not None:
            detected = boxes.detach().cpu().numpy()
            detected = sorted(detected, key=lambda b: float(b[4]*b[5]), reverse=True)[:max_dets]
            for b in detected:
                score = float(b[4]*b[5])
                x1, y1, x2, y2 = [float(v) for v in b[:4]/ratio]
                x1, x2 = np.clip([x1, x2], 0, w)
                y1, y2 = np.clip([y1, y2], 0, h)
                if x2 <= x1 or y2 <= y1:
                    continue
                predictions.append({'image_id': int(image_id), 'category_id': category_ids[0], 'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)], 'score': score})
                if score >= conf:
                    operating.append(((x1, y1, x2, y2), score))
        ann = coco.loadAnns(coco.getAnnIds(imgIds=[image_id], iscrowd=False))
        gts = []
        for a in ann:
            if a['category_id'] == category_ids[0] and not a.get('iscrowd', 0):
                x1, y1, bw, bh = a['bbox']
                gts.append((x1, y1, x1+bw, y1+bh))
        tp, fp, fn = match(gts, operating, match_iou)
        total_tp += tp; total_fp += fp; total_fn += fn
        gt, pred = len(gts), len(operating)
        records.append({'file_name': entry['file_name'], 'gt_count': gt, 'pred_count': pred, 'count_ratio_percent': round(pred/gt*100, 4) if gt else '', 'absolute_error': abs(pred-gt), 'tp': tp, 'fp': fp, 'fn': fn})
        print(f'{entry["file_name"]}: GT={gt} Pred={pred} TP={tp} FP={fp} FN={fn}', flush=True)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'count_per_image.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    (out / 'predictions_coco.json').write_text(json.dumps(predictions, indent=2), encoding='utf-8')
    gt_total = sum(r['gt_count'] for r in records)
    pred_total = sum(r['pred_count'] for r in records)
    metrics = {
        'evaluation_set': 'validation', 'images': len(records),
        'operating_confidence': conf, 'map_min_confidence': low_conf,
        'nms_iou': cfg['inference']['nms_iou'], 'matching_iou': match_iou,
        'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
        'precision_at_conf': total_tp/(total_tp+total_fp) if total_tp+total_fp else 0.0,
        'recall_at_conf': total_tp/(total_tp+total_fn) if total_tp+total_fn else 0.0,
        'gt_total': gt_total, 'pred_total': pred_total,
        'can_count_ratio_percent': 100*pred_total/gt_total if gt_total else None,
        'count_mae': sum(r['absolute_error'] for r in records)/len(records),
        'exact_count_accuracy': sum(r['gt_count'] == r['pred_count'] for r in records)/len(records),
    }
    if predictions:
        result = coco.loadRes(predictions)
        evaluator = COCOeval(coco, result, 'bbox')
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
        metrics['AP50'] = float(evaluator.stats[1]); metrics['mAP50_95'] = float(evaluator.stats[0])
    else:
        metrics['AP50'] = 0.0; metrics['mAP50_95'] = 0.0
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
    with (out / 'metrics.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['metric', 'value']); writer.writerows(metrics.items())
    print('\n--- FINAL VALIDATION METRICS ---')
    for key, value in metrics.items():
        print(f'{key}: {value}')
    print('Saved to:', out)


if __name__ == '__main__':
    main()
