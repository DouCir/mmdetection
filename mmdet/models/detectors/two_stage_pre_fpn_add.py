from .two_stage_mul import TwoStageDetectorMul
from ..registry import DETECTORS


@DETECTORS.register_module
class TwoStageDetectorPreFPNAdd(TwoStageDetectorMul):

    def extract_feat(self, img, img_t):
        # extract feature maps of RGB channel and Thermal channel respectively
        feats_rgb, feats_t = self.backbone(img, img_t)
        x = []
        for (r, t) in zip(feats_rgb, feats_t):
            temp = r + t  # elements-wise add in each feature maps of RGB images and thermal images
            x.append(temp)
        # build FPN based on fusion feature maps
        x = tuple(x)
        if self.with_neck:
            x = self.neck(x)
        return tuple(x)
