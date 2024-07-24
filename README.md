<div align="center">
  <img src="resources/mmdet-logo.png" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab website</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab platform</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>
</div>

## Installation

1.  Install MMEngine and MMCV using MIM.
    ```
    pip install -U openmim
    mim install mmengine
    mim install "mmcv>=2.0.0"
    ```

2.  Install MMDetection

    Case a: If you develop and run mmdet directly, install it from source:
    ```
    pip install -v -e .
    ```

    Case b: If you use mmdet as a dependency or third-party package, install it with MIM:
    ```
    mim install mmdet
    ```

## Customize Model

### Develop a new components

Here we show how to develop a new config for our model (e.g. Faster-RCNN)

1.  Create new file for our new model in `configs/faster-rcnn`. In this case we made a file named `faster-rcnn_r101_fpn_1x_bus.py` to train coco dataset for category `bus`.

2.  We will customize our Faster R-CNN model with the ResNet101 backbone. Firstly, we'll add the backbone to the file `faster-rcnn_r101_fpn_1x_bus.py` as follows:
    ```
    _base_ = './faster-rcnn_r50_fpn_1x_coco.py'
    
    model = dict(
      backbone=dict(
          depth=101,
          init_cfg=dict(type='Pretrained',
                        checkpoint='torchvision://resnet101')))
    ```

3.  Since we have our COCO dataset already filtered for the bus category, we need to specify the classes in the configuration as well. If we only use the configuration from MMDetection, it will not give us the correct classes for our dataset. We will add new code as follows:
     ```
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
     ``` 
4.  After adding the configuration for the dataset, we also update the code for the model that we've already created before, giving it the `roi_head` configuration.
    ```
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
    ```

    The full code will be displayed like this:
    ```
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
    ```

5.  If we want to change the `batch_size`, `max_epochs`, `val_interval`, or anything what you want, we can make a changes in the file `configs/_base_/schedules/scedule_1.py`.

## Training and Testing

To train a model with the new config, you can simply run
```
python tools/train.py configs/faster_rcnn/faster-rcnn_r101_fpn_1x_bus.py
```
Run the above training command, `wor_dirs/faster-rcnn_r101_fpn_1x_bu` folder will be automatically generated, the checkpoint file and the training config file will be saved in this folder.

### Training is resumed after the interruption

If you stop training, you can add `--resume` to the end of the training command and the program will automatically resume training with the latest weights file from `work_dirs`.
```
python tools/train.py configs/faster_rcnn/faster-rcnn_r101_fpn_1x_bus.py --resume
```

### Testing

To testing a model, you can simply run
```
python tools/test.py configs/faster_rcnn/faster-rcnn_r101_fpn_1x_bus.py \
                      work_dirs/faster-rcnn_r101_fpn_1x_bus/epoch_10.pth \
                     --show-dir faster_rcnn_r101_fpn_1x_results
```
Run the above test command, you can not only get the AP performance printed in the Training section, You can also automatically save the result images to the `work_dirs/faster_rcnn_r101_fpn_1x_results/{timestamp}/show_results` folder. 

## Convert Model

Before we convert the model, we must to install the `mmdeploy` module and library for the conversion. 

```
pip install mmdeploy
pip install onnx
pip install onnruntime==1.8.1
```

If you are using Docker devcontainer, once installed, you can use the following command to transform and deploy the trained model on the bus dataset with one click. However, if you are not using Docker devcontainer, please ensure that you have cloned the `mmdeploy` repository and modified the code inside the `tools/convert.py` file to refer to the path of the mmdeploy configuration.

```
python tools/convert.py
```

If we want to change the parameter of the conversion for the ONNX model, we can add a new file to create a new configuration. For example, if we want to change `max_output_boxes_per_class`, the first step is to create a new file named `tools/onnxruntime_dynamic.py`.  Then, we add the new code as follows:
```
_base_ = '../mmdeploy/configs/mmdet/detection/detection_onnxruntime_dynamic.py'

codebase_config = dict(
    post_processing=dict(
        max_output_boxes_per_class=500,
    ))
```

Don't forget that `_base_` refers to the mmdeploy configuration, so ensure that your path is appropriate for your mmdeploy path. Also, if you change other parameters, make sure to include the code for those parameters inside the file.

## Inference Model ONNX

Inference model ONNX using `onnxruntime`, you can simply run
```
python tools/infer.py --model=work_dirs/mmdeploy_models/mmdet/onnx/bus10.onnx \
                        --input=data/val_bus2024/000000443498.jpg \
                        --output=demo/detected_image.jpg
```