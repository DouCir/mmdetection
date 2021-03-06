from ..custom import CustomDataset
import os.path as osp
import numpy as np
from mmcv.parallel import DataContainer as DC
from ..utils import to_tensor, random_scale
import cv2
"""
Author: Yuan Yuan
Date: 2019/02/21
Description: This file defines a dataset which is used for pre-fineturing the Thermal branch.
             Data from CVC-09 theraml dataset and R channel of Caltech dataset.
"""

# dataset contains cvc-09 and the R channel of caltech
class ExtendedCvcDataset(CustomDataset):

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        # load image(thermal)
        img_t_path = osp.join(self.img_prefix, img_info['filename']).replace('visible', 'lwir')
        img_temp = cv2.imread(img_t_path)
        img_t = np.zeros((img_temp.shape[0], img_temp.shape[1], 3))
        if img_temp.shape[2] == 1:  # if the input image has only one channel,duplicate three times
            img_t[:, :, 0] = img_temp
            img_t[:, :, 1] = img_temp
            img_t[:, :, 2] = img_temp
        elif img_temp.shape[2] == 3:
            img_t[:, :, 0] = img_temp[:, :, 2]  # if the image has three channels, duplicate the R channel three times
            img_t[:, :, 1] = img_temp[:, :, 2]
            img_t[:, :, 2] = img_temp[:, :, 2]  # opencv : output BGR order
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
        img_t, img_shape, pad_shape, scale_factor = self.img_transform(
            img_t, img_scale, flip)
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
            img=DC(to_tensor(img_t), stack=True),
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
        return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        img_info = self.img_infos[idx]
        # load image(thermal)
        img_t_path = osp.join(self.img_prefix, img_info['filename']).replace('visible', 'lwir')
        img_temp = cv2.imread(img_t_path)
        img_t = np.zeros((img_temp.shape[0], img_temp.shape[1], 3))
        if img_temp.shape[2] == 1:  # if the input image has only one channel,duplicate three times
            img_t[:, :, 0] = img_temp
            img_t[:, :, 1] = img_temp
            img_t[:, :, 2] = img_temp
        elif img_temp.shape[2] == 3:
            img_t[:, :, 0] = img_temp[:, :, 2]  # if the image has three channels, duplicate the R channel three times
            img_t[:, :, 1] = img_temp[:, :, 2]
            img_t[:, :, 2] = img_temp[:, :, 2]  # opencv : output BGR order

        if self.proposals is not None:
            proposal = self.proposals[idx][:self.num_max_proposals]
            if not (proposal.shape[1] == 4 or proposal.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposal.shape))
        else:
            proposal = None

        def prepare_single(img_t, scale, flip, proposal=None):
            _img_t, img_shape, pad_shape, scale_factor = self.img_transform(
                img_t, scale, flip)
            _img_t = to_tensor(_img_t)
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
            return _img_t, _img_meta, _proposal

        imgs_t = []
        img_metas = []
        proposals = []
        for scale in self.img_scales:
            _img_t, _img_meta, _proposal = prepare_single(
                img_t, scale, False, proposal)
            imgs_t.append(_img_t)
            img_metas.append(DC(_img_meta, cpu_only=True))
            proposals.append(_proposal)
            if self.flip_ratio > 0:
                _img, _img_t, _img_meta, _proposal = prepare_single(img_t, scale, True, proposal)
                imgs_t.append(_img_t)
                img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)
        data = dict(img=imgs_t, img_meta=img_metas)
        if self.proposals is not None:
            data['proposals'] = proposals
        return data
