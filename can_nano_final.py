"""Official YOLOX-Nano Exp with YAML settings and dense-can annotation support.
Place this in the YOLOX repository root next to config_final.yaml.
The official Trainer requires CUDA; test on the GPU training computer.
"""
from exps.default.yolox_nano import Exp as NanoExp
from final_config import load_final_config


class Exp(NanoExp):
    def __init__(self):
        super().__init__()
        cfg = load_final_config()
        m, t, d, inf = cfg['model'], cfg['training'], cfg['data'], cfg['inference']
        self.exp_name = cfg['project']['experiment']
        self.output_dir = cfg['output']['root']
        self.seed = cfg['project']['seed']
        self.num_classes = m['classes']
        self.input_size = tuple(m['input_size'])
        self.test_size = tuple(m['input_size'])
        self.multiscale_range = 0
        self.random_size = (self.input_size[0] // 32, self.input_size[0] // 32)
        self.data_dir = d['root']
        self.train_ann = 'instances_train2017.json'
        self.val_ann = 'instances_val2017.json'
        self.max_epoch = t['epochs']
        self.data_num_workers = t['num_workers']
        self.basic_lr_per_img = t['base_learning_rate'] / t['batch_size']
        self.momentum = t['momentum']
        self.weight_decay = t['weight_decay']
        self.scheduler = t['scheduler']
        self.warmup_epochs = t['warmup_epochs']
        self.no_aug_epochs = t['no_aug_epochs']
        self.min_lr_ratio = t['min_lr_ratio']
        self.ema = t['ema']
        self.mosaic_prob = t['mosaic_probability']
        self.enable_mixup = t['mixup_enabled']
        self.mixup_prob = 1.0 if self.enable_mixup else 0.0
        self.flip_prob = t['flip_probability']
        self.hsv_prob = t['hsv_probability']
        self.eval_interval = t['eval_interval']
        self.print_interval = t['print_interval']
        self.save_history_ckpt = t['save_history']
        self.test_conf = cfg['evaluation']['map_min_confidence']
        self.nmsthre = inf['nms_iou']
        self.max_can_labels = t['max_labels']

    def get_dataset(self, cache=False, cache_type='ram'):
        from yolox.data import COCODataset, TrainTransform
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=self.max_can_labels,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
            ),
            cache=cache,
            cache_type=cache_type,
        )

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=None):
        import torch.distributed as dist
        from yolox.data import (
            TrainTransform, YoloBatchSampler, DataLoader,
            InfiniteSampler, MosaicDetection, worker_init_reset_seed,
        )
        from yolox.utils import wait_for_the_master
        
        if self.dataset is None:
            with wait_for_the_master():
                if cache_img not in (None, False):
                    raise ValueError(
                        "Image caching is not supported in this configuration"
                    )

                self.dataset = self.get_dataset(cache=False)

        self.dataset = MosaicDetection(
            dataset=self.dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=self.max_can_labels,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
            ),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )
        if is_distributed:
            batch_size //= dist.get_world_size()
        sampler = InfiniteSampler(len(self.dataset), seed=self.seed or 0)
        batch_sampler = YoloBatchSampler(
            sampler=sampler, batch_size=batch_size, drop_last=False,
            mosaic=not no_aug,
        )
        return DataLoader(
            self.dataset,
            batch_sampler=batch_sampler,
            num_workers=self.data_num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_reset_seed,
        )
