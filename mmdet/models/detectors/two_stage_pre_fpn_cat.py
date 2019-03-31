from .two_stage_mul import TwoStageDetectorMul
from ..registry import DETECTORS
import torch


@DETECTORS.register_module
class TwoStageDetectorPreFPNCat(TwoStageDetectorMul):

    def extract_feat(self, img, img_t):
        # extract feature maps of RGB channel and Thermal channel respectively
        feats_rgb, feats_t = self.backbone(img, img_t)
        x = []
        for (r, t) in zip(feats_rgb, feats_t):
            temp = torch.cat([r, t], 1)  # concatenate each feature maps of RGB images and thermal images
            x.append(temp)
        x = tuple(x)
        # build FPN based on fusion feature maps
        if self.with_neck:
            x = self.neck(x)
        return tuple(x)
