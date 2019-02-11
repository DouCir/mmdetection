import argparse

import torch
import mmcv
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from mmcv.parallel import scatter, collate, MMDataParallel

from mmdet import datasets
from mmdet.core import results2json, coco_eval, matlab_eval_MR
from mmdet.datasets import build_dataloader
from mmdet.models import build_detector, detectors
import os.path as osp
import numpy as np
import os
import sys

sys.setrecursionlimit(10 ** 8)


def single_test(model, data_loader, show=False):
    model.eval()
    results = []
    prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)

        # for test
        img_path = data_loader.dataset.img_infos[i]['filename']
        file_path = img_path.replace('images', 'res')
        file_path = file_path.replace('.jpg', '.txt')
        if os.path.exists(file_path):
            os.remove(file_path)
        os.mknod(file_path)
        np.savetxt(file_path, result[0])

        if show:
            model.module.show_result(data, result,
                                     data_loader.dataset.img_norm_cfg)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def _data_func(data, device_id):
    data = scatter(collate([data], samples_per_gpu=1), [device_id])[0]
    return dict(return_loss=False, rescale=True, **data)


def main():
    configs = ['../configs/caltech/faster_rcnn_v16_fpn_caltech_1x.py']

    for config in configs:
        # load dataset
        cfg = mmcv.Config.fromfile(config)
        cfg.model.pretrained = None
        cfg.data.test.test_mode = True
        dataset = obj_from_dict(cfg.data.val, datasets, dict(test_mode=True))
        # load model
        checkpoint_file = osp.join(cfg.work_dir, 'epoch_1.pth')
        model = build_detector(
            cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        load_checkpoint(model, checkpoint_file)
        model = MMDataParallel(model, device_ids=[0])
        # dataloader
        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            num_gpus=1,
            dist=False,
            shuffle=False)
        # outputs = single_test(model, data_loader, False)
        matlab_eval_MR()


if __name__ == '__main__':
    main()
