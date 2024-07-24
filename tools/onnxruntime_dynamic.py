_base_ = './onnx_config.py'

onnx_config = dict(output_names=['dets', 'labels'], input_shape=None)
codebase_config = dict(
    type='mmdet',
    task='ObjectDetection',
    model_type='end2end',
    post_processing=dict(
        score_threshold=0.05,
        confidence_threshold=0.005,  # for YOLOv3
        iou_threshold=0.5,
        max_output_boxes_per_class=500,
        pre_top_k=5000,
        keep_top_k=100,
        background_label_id=-1,
    ))

onnx_config = dict(
    dynamic_axes={
        'input': {
            0: 'batch',
            2: 'height',
            3: 'width'
        },
        'dets': {
            0: 'batch',
            1: 'num_dets',
        },
        'labels': {
            0: 'batch',
            1: 'num_dets',
        },
    }, )

backend_config = dict(type='onnxruntime')


