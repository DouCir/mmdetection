from .two_stage_pre_fpn_add import TwoStageDetectorPreFPNAdd
from ..registry import DETECTORS


@DETECTORS.register_module
class FasterRCNNMulPreFPNAdd(TwoStageDetectorPreFPNAdd):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 upper_neck=None,
                 pretrained=None):
        super(FasterRCNNMulPreFPNAdd, self).__init__(
            backbone=backbone,
            neck=neck,
            upper_neck=upper_neck,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
