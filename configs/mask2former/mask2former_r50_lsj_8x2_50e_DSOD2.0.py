_base_ = ['./mask2former_r50_lsj_8x2_50e_coco-panoptic.py']
num_things_classes = 1
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes
norm_cfg = dict(type='SyncABN', requires_grad=True)
model = dict(
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    # backbone=dict(
    #     type='ResNetDNL',
    #     depth=50,
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3),
    #     frozen_stages=1,
    #     style='pytorch',
    #     nlgcb=dict(ratio=1. / 4., downsample=False, whiten_type=['channel'], temp=0.05, with_gc=True, use_out=False, out_bn=False),
    #     stage_with_nlgcb=[[], [], [-2], [-2, -1, 0]],
    #     norm_eval=False,
    #     norm_cfg=norm_cfg,
    #     non_inplace=True),
    panoptic_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_cls=dict(class_weight=[1.0] * num_classes + [0.1])),
    panoptic_fusion_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes),
    test_cfg=dict(panoptic_on=False))

# dataset settings
# image_size = (1024, 1024)
image_size = (512, 384)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
pad_cfg = dict(img=(128, 128, 128), masks=0, seg=255)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    # large scale jittering
    dict(
        type='Resize',
        img_scale=image_size,
        ratio_range=(0.1, 2.0),
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=image_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='Pad', size=image_size, pad_val=pad_cfg),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle', img_to_float=True),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(1333, 800),
        img_scale=(640, 480),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Pad', size_divisor=32, pad_val=pad_cfg),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
dataset_type = 'CocoDataset'
classes = ('surface',)
data_root = '/data/home/wangxu/datasets/DSOD2.0/VDSOD2.0/'
data = dict(
    _delete_=True,
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/carton_val_coco_annotations.json',
        img_prefix=data_root + 'val/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/carton_val_coco_annotations.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/carton_val_coco_annotations.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])
