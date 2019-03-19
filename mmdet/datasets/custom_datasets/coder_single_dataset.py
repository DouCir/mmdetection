from ..custom import CustomDataset
import os.path as osp

import mmcv
import numpy as np
from ..utils import to_tensor, random_scale
from ..transforms import ImageTransform
import cv2


class CoderSingleDataset(CustomDataset):
    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 img_norm_cfg_t,
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 with_mask=False,
                 with_crowd=True,
                 with_label=True,
                 test_mode=False):
        self.img_norm_cfg_t = img_norm_cfg_t
        # transforms
        self.img_transform_t = ImageTransform(
            size_divisor=size_divisor, **self.img_norm_cfg_t)
        super(CoderSingleDataset, self).__init__(
            ann_file=ann_file,
            img_prefix=img_prefix,
            img_scale=img_scale,
            img_norm_cfg=img_norm_cfg,
            size_divisor=size_divisor,
            proposal_file=proposal_file,
            num_max_proposals=num_max_proposals,
            flip_ratio=flip_ratio,
            with_mask=with_mask,
            with_crowd=with_crowd,
            with_label=with_label,
            test_mode=test_mode)

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        flag = img_info['flag']
        # load image(rgb)
        img_temp = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        if img_temp.shape[2] == 1:
            img = np.zeros((img_temp.shape[0], img_temp.shape[1], 3))
            img[:, :, 0] = img_temp
            img[:, :, 1] = img_temp
            img[:, :, 2] = img_temp
        else:
            img = img_temp
        # load image(thermal)
        img_t_path = osp.join(self.img_prefix, img_info['filename']).replace('visible', 'lwir')
        img_t = cv2.imread(img_t_path)  # three channels,??? img_t[:,:,0]==img_t[:,:,2]!= img_t[:,:,1]
        img_t[:, :, 1] = img_t[:, :, 0]
        if img_t[0].max() > 140:
            a = 10
        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        img_scale = random_scale(self.img_scales)  # sample a scale
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, img_scale, flip)
        img_t, img_shape_t, pad_shape_t, scale_factor_t = self.img_transform_t(
            img_t, img_scale, flip)

        data = dict(
            img_rgb_out=to_tensor(img),
            img_thermal_out=to_tensor(img_t))
        # default multispectral
        ori_shape = (img_info['height'], img_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip)

        data['img_meta'] = img_meta
        data['img_rgb_in'] = to_tensor(img)
        data['img_thermal_in'] = to_tensor(img_t)
        return data

    def prepare_test_img(self, idx):
        return self.prepare_train_img(idx)
