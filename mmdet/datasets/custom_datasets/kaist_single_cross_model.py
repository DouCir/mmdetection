from .kaist_single import KaistRGBDataset
from mmcv.runner import load_checkpoint
import cv2
import os.path as osp
import numpy as np
from mmcv.parallel import DataContainer as DC
from ..utils import to_tensor, random_scale
from ..transforms import (ImageTransform, BboxTransform, MaskTransform,
                          Numpy2Tensor)
import torch


# thermal cross model dataset
class KaistThermalCross(KaistRGBDataset):
    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 img_norm_cross_rgb,
                 img_norm_cross_t,
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 with_mask=False,
                 with_crowd=True,
                 with_label=True,
                 test_mode=False):
        super(KaistThermalCross, self).__init__(
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
        self.img_norm_cross_rgb = img_norm_cross_rgb
        self.img_norm_cross_t = img_norm_cross_t
        # transforms
        self.img_transform_cross_rgb = ImageTransform(
            size_divisor=size_divisor, **self.img_norm_cross_rgb)
        self.img_transform_cross_t = ImageTransform(
            size_divisor=size_divisor, **self.img_norm_cross_t)

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        # load image(thermal)
        img_path = osp.join(self.img_prefix, img_info['filename']).replace('visible', 'lwir')
        img_ori = cv2.imread(img_path)
        img_ori[:, :, 1] = img_ori[:, :, 0];
        # load proposals if necessary
        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]
            # TODO: Handle empty proposals properly. Currently images with
            # no proposals are just ignored, but they can be used for
            # training in concept.
            if len(proposals) == 0:
                return None
            if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 5:
                scores = proposals[:, 4, None]
                proposals = proposals[:, :4]
            else:
                scores = None

        ann = self.get_ann_info(idx)
        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']
        if self.with_crowd:
            gt_bboxes_ignore = ann['bboxes_ignore']

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0:
            return None

        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        img_scale = random_scale(self.img_scales)  # sample a scale
        # for thermal images
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img_ori, img_scale, flip)
        if self.proposals is not None:
            proposals = self.bbox_transform(proposals, img_shape, scale_factor,
                                            flip)
            proposals = np.hstack(
                [proposals, scores]) if scores is not None else proposals
        gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
                                        flip)
        if self.with_crowd:
            gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                                   scale_factor, flip)
        if self.with_mask:
            gt_masks = self.mask_transform(ann['masks'], pad_shape,
                                           scale_factor, flip)

        ori_shape = (img_info['height'], img_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip)
        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_bboxes=DC(to_tensor(gt_bboxes)))

        if self.proposals is not None:
            data['proposals'] = DC(to_tensor(proposals))
        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
        if self.with_crowd:
            data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
        if self.with_mask:
            data['gt_masks'] = DC(gt_masks, cpu_only=True)

        # prepare images used by autoencoder
        img_norm_thermal, _, _, _ = self.img_transform_cross_t(img_ori, img_scale, flip)
        img_norm_rgb = np.zeros(img_norm_thermal.shape, dtype='float32')
        data['img_norm_rgb'] = to_tensor(img_norm_rgb)
        data['img_norm_thermal'] = to_tensor(img_norm_thermal)
        return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        img_info = self.img_infos[idx]
        # load image(thermal)
        img_path = osp.join(self.img_prefix, img_info['filename']).replace('visible', 'lwir')
        img_ori = cv2.imread(img_path)
        img_ori[:, :, 1] = img_ori[:, :, 0]
        if self.proposals is not None:
            proposal = self.proposals[idx][:self.num_max_proposals]
            if not (proposal.shape[1] == 4 or proposal.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposal.shape))
        else:
            proposal = None

        def prepare_single(img_ori, scale, flip, proposal=None):
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img_ori, scale, flip)
            _img = to_tensor(_img)
            _img_meta = dict(
                ori_shape=(img_info['height'], img_info['width'], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip)
            if proposal is not None:
                if proposal.shape[1] == 5:
                    score = proposal[:, 4, None]
                    proposal = proposal[:, :4]
                else:
                    score = None
                _proposal = self.bbox_transform(proposal, img_shape,
                                                scale_factor, flip)
                _proposal = np.hstack(
                    [_proposal, score]) if score is not None else _proposal
                _proposal = to_tensor(_proposal)
            else:
                _proposal = None
            return _img, _img_meta, _proposal

        imgs = []
        img_metas = []
        proposals = []
        for scale in self.img_scales:
            _img, _img_meta, _proposal = prepare_single(
                img_ori, scale, False, proposal)
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            proposals.append(_proposal)
            if self.flip_ratio > 0:
                _img, _img_meta, _proposal = prepare_single(img_ori, scale, True, proposal)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)
        data = dict(img=imgs, img_meta=img_metas)
        if self.proposals is not None:
            data['proposals'] = proposals
        # prepare images used by autoencoder
        img_norm_rgbs = []
        img_norm_thermals = []
        img_norm_thermal, _, _, _ = self.img_transform_cross_t(img_ori, self.img_scales[0], False)
        img_norm_rgb = np.zeros(img_norm_thermal.shape, dtype='float32')
        img_norm_rgbs.append(img_norm_rgb)
        img_norm_thermals.append(img_norm_thermal)
        data['img_norm_rgb'] = img_norm_rgbs
        data['img_norm_thermal'] = img_norm_thermals
        return data


# RGB cross model dataset
class KaistRGBCross(KaistRGBDataset):
    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 img_norm_cross_rgb,
                 img_norm_cross_t,
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 with_mask=False,
                 with_crowd=True,
                 with_label=True,
                 test_mode=False):
        super(KaistRGBCross, self).__init__(
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
        self.img_norm_cross_rgb = img_norm_cross_rgb
        self.img_norm_cross_t = img_norm_cross_t
        # transforms
        self.img_transform_cross_rgb = ImageTransform(
            size_divisor=size_divisor, **self.img_norm_cross_rgb)
        self.img_transform_cross_t = ImageTransform(
            size_divisor=size_divisor, **self.img_norm_cross_t)

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        # load image(thermal)
        img_path = osp.join(self.img_prefix, img_info['filename'])
        img_ori = cv2.imread(img_path)
        # load proposals if necessary
        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]
            # TODO: Handle empty proposals properly. Currently images with
            # no proposals are just ignored, but they can be used for
            # training in concept.
            if len(proposals) == 0:
                return None
            if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 5:
                scores = proposals[:, 4, None]
                proposals = proposals[:, :4]
            else:
                scores = None

        ann = self.get_ann_info(idx)
        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']
        if self.with_crowd:
            gt_bboxes_ignore = ann['bboxes_ignore']

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0:
            return None

        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        img_scale = random_scale(self.img_scales)  # sample a scale
        # for thermal images
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img_ori, img_scale, flip)
        if self.proposals is not None:
            proposals = self.bbox_transform(proposals, img_shape, scale_factor,
                                            flip)
            proposals = np.hstack(
                [proposals, scores]) if scores is not None else proposals
        gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
                                        flip)
        if self.with_crowd:
            gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                                   scale_factor, flip)
        if self.with_mask:
            gt_masks = self.mask_transform(ann['masks'], pad_shape,
                                           scale_factor, flip)

        ori_shape = (img_info['height'], img_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip)
        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_bboxes=DC(to_tensor(gt_bboxes)))

        if self.proposals is not None:
            data['proposals'] = DC(to_tensor(proposals))
        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
        if self.with_crowd:
            data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
        if self.with_mask:
            data['gt_masks'] = DC(gt_masks, cpu_only=True)

        # prepare images used by autoencoder
        img_norm_rgb, _, _, _ = self.img_transform_cross_rgb(img_ori, img_scale, flip)
        img_norm_thermal = np.zeros(img_norm_rgb.shape, dtype='float32')
        data['img_norm_rgb'] = to_tensor(img_norm_rgb)
        data['img_norm_thermal'] = to_tensor(img_norm_thermal)
        return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        img_info = self.img_infos[idx]
        # load image(thermal)
        img_path = osp.join(self.img_prefix, img_info['filename'])
        img_ori = cv2.imread(img_path)
        if self.proposals is not None:
            proposal = self.proposals[idx][:self.num_max_proposals]
            if not (proposal.shape[1] == 4 or proposal.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposal.shape))
        else:
            proposal = None

        def prepare_single(img_ori, scale, flip, proposal=None):
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img_ori, scale, flip)
            _img = to_tensor(_img)
            _img_meta = dict(
                ori_shape=(img_info['height'], img_info['width'], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip)
            if proposal is not None:
                if proposal.shape[1] == 5:
                    score = proposal[:, 4, None]
                    proposal = proposal[:, :4]
                else:
                    score = None
                _proposal = self.bbox_transform(proposal, img_shape,
                                                scale_factor, flip)
                _proposal = np.hstack(
                    [_proposal, score]) if score is not None else _proposal
                _proposal = to_tensor(_proposal)
            else:
                _proposal = None
            return _img, _img_meta, _proposal

        imgs = []
        img_metas = []
        proposals = []
        for scale in self.img_scales:
            _img, _img_meta, _proposal = prepare_single(
                img_ori, scale, False, proposal)
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            proposals.append(_proposal)
            if self.flip_ratio > 0:
                _img, _img_meta, _proposal = prepare_single(img_ori, scale, True, proposal)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)
        data = dict(img=imgs, img_meta=img_metas)
        if self.proposals is not None:
            data['proposals'] = proposals
        # prepare images used by autoencoder
        img_norm_rgbs = []
        img_norm_thermals = []
        img_norm_rgb, _, _, _ = self.img_transform_cross_rgb(img_ori, self.img_scales[0], False)
        img_norm_thermal = np.zeros(img_norm_rgb.shape, dtype='float32')
        img_norm_rgbs.append(img_norm_rgb)
        img_norm_thermals.append(img_norm_thermal)
        data['img_norm_rgb'] = img_norm_rgbs
        data['img_norm_thermal'] = img_norm_thermals
        return data
