_base_ = [
    '../_base_/models/resnet50.py',          # base model
    '../_base_/datasets/cifar100_bs32.py',  # dataset config
    '../_base_/schedules/sgd_100e.py',      # schedule: 100 epochs SGD
    '../_base_/default_runtime.py'           # runtime defaults
]

# Override dataset paths here — use placeholder 'path/to/cifar100' that will be replaced by your bash script
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type='CIFAR100',
        data_prefix='path/to/cifar100',
        pipeline=[
            dict(type='RandomCrop', size=32, padding=4),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(type='Normalize', mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
            dict(type='ToTensor'),
        ],
    ),
    val=dict(
        type='CIFAR100',
        data_prefix='path/to/cifar100',
        pipeline=[
            dict(type='Normalize', mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
            dict(type='ToTensor'),
        ],
    ),
)

# Load from checkpoint: None means training from scratch; replace if you have a pretrained checkpoint
load_from = None

# Adjust model head for CIFAR100 classes
model = dict(
    head=dict(
        num_classes=100
    )
)
