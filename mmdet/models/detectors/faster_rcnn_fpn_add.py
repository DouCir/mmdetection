from .two_stage_fpn_add import TwoStageDetectorFPNAdd
from ..registry import DETECTORS


@DETECTORS.register_module
class FasterRCNNMulFPNAdd(TwoStageDetectorFPNAdd):

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
        super(FasterRCNNMulFPNAdd, self).__init__(
            backbone=backbone,
            neck=neck,
            upper_neck=upper_neck,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
