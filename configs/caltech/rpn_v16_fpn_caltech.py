# model settings
model = dict(
    type='RPN',
    pretrained='/media/ser606/Data/DoubleCircle/model/vgg16-397923af.pth',
    backbone=dict(
        type='VGG',
        depth=16,
        num_stages=5,
        out_indices=[1, 2, 3, 4],
        frozen_stages=1),
    neck=dict(
        type='FPN',
        in_channels=[128, 256, 512, 512],
        out_channels=256,
        num_outs=4,
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8, 10, 12, 14],
        anchor_ratios=[1 / 0.5, 1.0],
        anchor_strides=[4, 8, 16, 32],
        anchor_base_sizes=[4, 8, 16, 32],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        use_sigmoid_cls=True))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=128,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False,
            pos_balance_sampling=False,
            neg_balance_thr=0),
        allowed_border=0,
        pos_weight=-1,
        smoothl1_beta=1.0,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=20000,
        nms_post=20000,
        max_num=40,
        nms_thr=0.5,
        min_bbox_size=0))
# dataset settings
dataset_type = 'CaltechDataset'
data_root = '/media/ser606/Data/DoubleCircle/datasets/Caltech/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=6,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations-pkl/train-all.pkl',
        img_prefix=data_root + 'images/',
        img_scale=1.6,
        img_norm_cfg=img_norm_cfg,
        size_divisor=None,
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=False,
        with_label=False),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations-pkl/test-all.pkl',
        img_prefix=data_root + 'images/',
        img_scale=1.6,
        img_norm_cfg=img_norm_cfg,
        size_divisor=None,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=False),
    test=dict(
        type='CocoDataset',
        ann_file=data_root + 'annotations-json/test-all.json',
        img_prefix=data_root + 'images/',
        img_scale=1.6,
        img_norm_cfg=img_norm_cfg,
        size_divisor=None,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
# runner configs
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=5000,
    warmup_ratio=1.0 / 3,
    step=[8])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=1000,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 20
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '../../work_dirs/rpn_vgg16_fpn_caltech'
load_from = None
resume_from = None
workflow = [('train', 1)]
