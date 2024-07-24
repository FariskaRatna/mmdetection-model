_base_ = './faster-rcnn_r50_fpn_1x_coco.py'

dataset_type = 'CocoDataset'
classes = ('bus',)
data_root='data/'

backend_args = None

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='train_bus2024.json',
        data_prefix=dict(img='train_bus2024/')
        )
    )

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='val_bus2024.json',
        data_prefix=dict(img='val_bus2024/')
        )
    )

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='val_bus2024.json',
        data_prefix=dict(img='val_bus2024/')
        )
    )

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val_bus2024.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            num_classes=1
        )
    ))
