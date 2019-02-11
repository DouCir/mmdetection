from ..custom import CustomDataset


class CaltechDataset(CustomDataset):

    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 with_mask=False,
                 with_crowd=True,
                 with_label=True,
                 test_mode=False):
        super(CaltechDataset, self).__init__(
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
