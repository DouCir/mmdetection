from __future__ import division

from mmcv import Config
from mmcv.runner import obj_from_dict
from mmdet import datasets, __version__
from mmdet.apis import (train_detector, get_root_logger)
from mmdet.models import build_detector
import os
import os.path as osp
import getpass

"""
Author:Yuan Yuan
Date:2019/02/11
Description: This script is used to train detectors with config files.
"""


def main():

    configs = \
        [
            # '../../configs/caltech/faster_rcnn_r50_fpn_caltech.py',
            # '../../configs/caltech/faster_rcnn_r50_c4_caltech.py'

            # '../../configs/kaist/faster_rcnn_r50_c4_rgb_kaist.py',
            # '../../configs/kaist/faster_rcnn_r50_fpn_rgb_kaist.py',
            # '../../configs/kaist/faster_rcnn_r50_c4_thermal_kaist.py',
            # '../../configs/kaist/faster_rcnn_r50_fpn_thermal_kaist.py',

            # '../../configs/kaist/faster_rcnn_v16_c5_rgb_kaist.py',
            # '../../configs/kaist/faster_rcnn_v16_fpn_rgb_kaist.py',
            # '../../configs/kaist/faster_rcnn_v16_c5_thermal_kaist.py',
            # '../../configs/kaist/faster_rcnn_v16_fpn_thermal_kaist.py',

            # '../../configs/kaist/mul_faster_rcnn_v16_fpn_cat_kaist.py',
            #
            # '../../configs/kaist/mul_faster_rcnn_r50_c4_add_kaist.py',
            # '../../configs/kaist/mul_faster_rcnn_r50_fpn_add_kaist.py',
            # '../../configs/kaist/mul_faster_rcnn_v16_c5_add_kaist.py',
            # '../../configs/kaist/mul_faster_rcnn_v16_fpn_add_kaist.py',

            '../../configs/kaist-cross/cross_mul_faster_rcnn_r50_fpn_cat_kaist.py',
            '../../configs/kaist-cross/cross_mul_faster_rcnn_v16_fpn_cat_kaist.py'

        ]


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

        username = getpass.getuser()
        temp_file = '/media/' + username + '/Data/DoubleCircle/temp/temp.txt'
        fo = open(temp_file, 'w+')
        str_write = cfg.work_dir.replace('../..',
                                         ('/media/'+username+'/Data/DoubleCircle/project/mmdetection/mmdetection'))
        fo.write(str_write)
        fo.close()

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
