
from pathlib import Path

import cv2
import torch

from can_nano import Exp
from yolox.data.data_augment import ValTransform
from yolox.utils import postprocess


# 실행 설정
MODEL_PATH = "training_outputs/can_nano_finetuned.pth"
IMAGE_DIR = Path("datasets/CanCOCO/val2017")
OUTPUT_DIR = Path("prediction_results")

CONF_THRESHOLD = 0.50
NMS_THRESHOLD = 0.45
MAX_IMAGES = 5


def main():
    torch.set_num_threads(2)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 학습된 캔 탐지 모델 불러오기
    exp = Exp()
    model = exp.get_model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model"])
    model.eval()

    # 2. 학습과 동일한 입력 크기로 전처리
    transform = ValTransform(legacy=False)

    image_paths = sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in
        {".jpg", ".jpeg", ".png", ".bmp"}
    )[:MAX_IMAGES]

    if not image_paths:
        raise RuntimeError("검사할 이미지가 없습니다.")

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"읽기 실패: {image_path}")
            continue

        height, width = image.shape[:2]

        ratio = min(
            exp.test_size[0] / height,
            exp.test_size[1] / width,
        )

        # 3. 비율 유지 축소 + 패딩
        processed, _ = transform(
            image, None, exp.test_size
        )

        tensor = (
            torch.from_numpy(processed)
            .unsqueeze(0)
            .float()
        )

        # 4. 객체 탐지
        with torch.no_grad():
            outputs = model(tensor)
            predictions = postprocess(
                outputs,
                num_classes=1,
                conf_thre=CONF_THRESHOLD,
                nms_thre=NMS_THRESHOLD,
                class_agnostic=True,
            )[0]

        result_image = image.copy()
        count = 0

        # 5. 탐지된 캔에 박스 그리기
        if predictions is not None:
            predictions = predictions.cpu()

            for detection in predictions:
                x1, y1, x2, y2 = (
                    detection[:4].numpy() / ratio
                )

                confidence = (
                    detection[4] * detection[5]
                ).item()

                x1 = max(0, min(int(x1), width - 1))
                y1 = max(0, min(int(y1), height - 1))
                x2 = max(0, min(int(x2), width - 1))
                y2 = max(0, min(int(y2), height - 1))

                cv2.rectangle(
                    result_image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                count += 1

        # 6. 캔 개수 표시
        cv2.putText(
            result_image,
            f"Detected cans: {count}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # 7. 결과 이미지 저장
        output_path = OUTPUT_DIR / image_path.name
        cv2.imwrite(str(output_path), result_image)

        print(
            f"{image_path.name} "
            f"| 예측 캔: {count}개 "
            f"| 저장: {output_path}"
        )

    print("탐지 테스트 완료!")


if __name__ == "__main__":
    main()
