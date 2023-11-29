_base_ = ['./mask2former_r50_8xb2-lsj-50e_coco-panoptic.py']

num_things_classes = 1
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes
image_size = (256, 256)

batch_augments = [
    dict(
        type='BatchFixedSizePad',
        size=image_size,
        img_pad_value=0,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=False)
]
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=False,
    batch_augments=batch_augments)
model = dict(
    data_preprocessor=data_preprocessor,
    # backbone=dict(
    #     type='ResNetDNL',
    #     depth=50,
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3),
    #     frozen_stages=-1,
    #     norm_cfg=dict(type='BN', requires_grad=False),
    #     norm_eval=True,
    #     style='pytorch',
    #     init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    backbone=dict(
        type='ResNetDNL',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',
        nlgcb=dict(ratio=1. / 4., downsample=False, whiten_type=['channel'], temp=0.05, with_gc=True, use_out=False, out_bn=False),
        stage_with_nlgcb=[[], [], [-2], [-2, -1, 0]],
        norm_eval=False,
        norm_cfg=dict(type='BN', requires_grad=False),
        # norm_cfg=dict(type='SyncABN', requires_grad=True),
        non_inplace=True,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    panoptic_head=dict(
        type='Mask2FormerHead',
        in_channels=[256, 512, 1024, 2048],  # pass to pixel_decoder inside
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        num_queries=100,
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type='MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(  # DeformableDetrTransformerEncoder
                num_layers=6,
                layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
                    self_attn_cfg=dict(  # MultiScaleDeformableAttention
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        dropout=0.0,
                        batch_first=True),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)))),
            positional_encoding=dict(num_feats=128, normalize=True)),
        enforce_decoder_input_project=False,
        positional_encoding=dict(num_feats=128, normalize=True),
        transformer_decoder=dict(  # Mask2FormerTransformerDecoder
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(  # Mask2FormerTransformerDecoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.0,
                    batch_first=True),
                cross_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.0,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type='ReLU', inplace=True))),
            init_cfg=None),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0)),
    panoptic_fusion_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes),
    test_cfg=dict(panoptic_on=False))

# dataset settings
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    # large scale jittering
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.1, 2.0),
        resize_type='Resize',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=image_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=(640, 480), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# dataset_type = 'CocoDataset'
# data_root = 'data/coco/'
# dataset settings
dataset_type = 'CocoDataset'
data_root = '/data/home/wangxu/datasets/DSOD2.0/10/'
# data_root_test = '/data/home/wangxu/datasets/SCD/LSCD/coco_style_oneclass/'
metainfo = {
  'classes': ('surface', ),
  'palette': [
        (220, 20, 60),
    ]
}

backend_args = None

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='output_carton_train/coco_data/coco_annotations.json',
        data_prefix=dict(img='output_surface_train/coco_data/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='output_carton_train/coco_data/coco_annotations.json',
        data_prefix=dict(img='output_surface_train/coco_data/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    ann_file=data_root + 'output_carton_train/coco_data/coco_annotations.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args={{_base_.backend_args}})
test_evaluator = val_evaluator


# learning policy
max_iters = 1000
param_scheduler = dict(
    type='MultiStepLR',
    begin=0,
    end=max_iters,
    by_epoch=False,
    milestones=[327778, 355092],
    gamma=0.1)

# Before 365001th iteration, we do evaluation every 5000 iterations.
# After 365000th iteration, we do evaluation every 368750 iterations,
# which means that we do evaluation at the end of training.
interval = 100
dynamic_intervals = [(max_iters // interval * interval + 1, max_iters)]
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=interval,
    dynamic_intervals=dynamic_intervals)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        save_last=True,
        max_keep_ckpts=3,
        interval=interval))
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)