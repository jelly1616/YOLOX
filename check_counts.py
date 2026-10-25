
from pathlib import Path
from pycocotools.coco import COCO

# Validation 정답 데이터
coco = COCO(
    "datasets/CanCOCO/annotations/instances_val2017.json"
)

# 방금 탐지한 5개 이미지
filenames = [
    "000084.jpg",
    "000095.jpg",
    "000101.jpg",
    "000104.jpg",
    "000108.jpg",
]

predicted_counts = [63, 38, 32, 43, 51]

images_by_name = {
    img["file_name"]: img["id"]
    for img in coco.dataset["images"]
}

print("이미지 | 정답 | 예측 | Count Ratio | 절대 오차")
print("-" * 60)

for filename, pred in zip(filenames, predicted_counts):
    image_id = images_by_name.get(filename)

    if image_id is None:
        print(f"{filename}: 정답 이미지 없음")
        continue

    ann_ids = coco.getAnnIds(imgIds=[image_id])
    gt = len(ann_ids)

    ratio = pred / gt * 100 if gt > 0 else None
    error = abs(gt - pred)

    ratio_text = (
        f"{ratio:.2f}%" if ratio is not None else "N/A"
    )

    print(
        f"{filename} | {gt} | {pred} | "
        f"{ratio_text} | {error}"
    )
