checkpoint_config = dict(interval=1)
data = dict(
    samples_per_gpu=32,
    test=dict(
        ann_file='data/imagenet/meta/val.txt',
        data_prefix='data/imagenet/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(size=(
                256,
                -1,
            ), type='Resize'),
            dict(crop_size=224, type='CenterCrop'),
            dict(
                mean=[
                    123.675,
                    116.28,
                    103.53,
                ],
                std=[
                    58.395,
                    57.12,
                    57.375,
                ],
                to_rgb=True,
                type='Normalize'),
            dict(keys=[
                'img',
            ], type='ImageToTensor'),
            dict(keys=[
                'img',
            ], type='Collect'),
        ],
        type='ImageNet'),
    train=dict(
        data_prefix='data/imagenet/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(size=224, type='RandomResizedCrop'),
            dict(direction='horizontal', flip_prob=0.5, type='RandomFlip'),
            dict(
                mean=[
                    123.675,
                    116.28,
                    103.53,
                ],
                std=[
                    58.395,
                    57.12,
                    57.375,
                ],
                to_rgb=True,
                type='Normalize'),
            dict(keys=[
                'img',
            ], type='ImageToTensor'),
            dict(keys=[
                'gt_label',
            ], type='ToTensor'),
            dict(keys=[
                'img',
                'gt_label',
            ], type='Collect'),
        ],
        type='ImageNet'),
    val=dict(
        ann_file='data/imagenet/meta/val.txt',
        data_prefix='data/imagenet/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(size=(
                256,
                -1,
            ), type='Resize'),
            dict(crop_size=224, type='CenterCrop'),
            dict(
                mean=[
                    123.675,
                    116.28,
                    103.53,
                ],
                std=[
                    58.395,
                    57.12,
                    57.375,
                ],
                to_rgb=True,
                type='Normalize'),
            dict(keys=[
                'img',
            ], type='ImageToTensor'),
            dict(keys=[
                'img',
            ], type='Collect'),
        ],
        type='ImageNet'),
    workers_per_gpu=2)
dataset_type = 'ImageNet'
dist_params = dict(backend='nccl')
evaluation = dict(interval=1, metric='accuracy')
img_norm_cfg = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
load_from = None
log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
    ], interval=100)
log_level = 'INFO'
lr_config = dict(
    policy='step', step=[
        30,
        60,
        90,
    ])
model = dict(
    backbone=dict(
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        type='ResNet'),
    head=dict(
        in_channels=2048,
        loss=dict(loss_weight=1.0, type='CrossEntropyLoss'),
        num_classes=1000,
        topk=(
            1,
            5,
        ),
        type='LinearClsHead'),
    neck=dict(type='GlobalAveragePooling'),
    type='ImageClassifier')
optimizer = dict(lr=0.1, momentum=0.9, type='SGD', weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
resume_from = None
runner = dict(max_epochs=100, type='EpochBasedRunner')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(size=(
        256,
        -1,
    ), type='Resize'),
    dict(crop_size=224, type='CenterCrop'),
    dict(
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        to_rgb=True,
        type='Normalize'),
    dict(keys=[
        'img',
    ], type='ImageToTensor'),
    dict(keys=[
        'img',
    ], type='Collect'),
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(size=224, type='RandomResizedCrop'),
    dict(direction='horizontal', flip_prob=0.5, type='RandomFlip'),
    dict(
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        to_rgb=True,
        type='Normalize'),
    dict(keys=[
        'img',
    ], type='ImageToTensor'),
    dict(keys=[
        'gt_label',
    ], type='ToTensor'),
    dict(keys=[
        'img',
        'gt_label',
    ], type='Collect'),
]
workflow = [
    (
        'train',
        1,
    ),
]
