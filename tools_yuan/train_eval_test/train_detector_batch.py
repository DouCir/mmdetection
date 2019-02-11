from __future__ import division

from mmcv import Config
from mmcv.runner import obj_from_dict

from mmdet import datasets, __version__
from mmdet.apis import (train_detector, init_dist, get_root_logger,
                        set_random_seed)
from mmdet.models import build_detector
import os


def main():

    configs = ['../../configs/caltech/rpn_vgg16_fpn_caltech.py']

    for config in configs:
        # load dataset
        cfg = Config.fromfile(config)
        cfg.gpus = 1
        if not os.path.exists(cfg.work_dir):
            os.mkdir(cfg.work_dir)
        if cfg.checkpoint_config is not None:
            # save mmdet version in checkpoints as meta data
            cfg.checkpoint_config.meta = dict(
                mmdet_version=__version__, config=cfg.text)
        distributed = False
        # init logger before other steps
        logger = get_root_logger(cfg.log_level)
        logger.info('Distributed training: {}'.format(distributed))
        # build model
        model = build_detector(
            cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
        # create datasets used for train and validation
        train_dataset = obj_from_dict(cfg.data.train, datasets)
        # train a detector
        train_detector(
            model,
            train_dataset,
            cfg,
            distributed=distributed,
            validate=True,
            logger=logger)


if __name__ == '__main__':
    main()