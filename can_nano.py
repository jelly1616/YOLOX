
from exps.default.yolox_nano import Exp as NanoExp


class Exp(NanoExp):
    def __init__(self):
        super().__init__()

        # 탐지 대상: 캔 1종류
        self.num_classes = 1

        # 이미지 입력 크기
        self.input_size = (416, 416)
        self.test_size = (416, 416)

        # 학습 설정 (초기 개발용)
        self.max_epoch = 3
        self.warmup_epochs = 0
        self.no_aug_epochs = 0
        self.eval_interval = 1

        # 초기에는 입력 해상도 고정
        self.multiscale_range = 0
        self.random_size = (13, 13)

        # 데이터셋 설정 (나중에 사용할 경로)
        self.data_dir = "datasets/CanCOCO"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"

        # 실험 이름
        self.exp_name = "can_yolox_nano"
