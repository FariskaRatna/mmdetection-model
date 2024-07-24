from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
import torch

deploy_cfg = 'mmdeploy/configs/mmdet/detection/detection_onnxruntime_dynamic.py'
model_cfg = 'configs/faster_rcnn/faster-rcnn_r101_fpn_1x_bus.py'
device = 'cpu'
backend_model = ['work_dirs/mmdeploy_models/mmdet/onnx/bus10.onnx']
image = 'data/val_bus2024/000000001584.jpg'

# read deploy_cfg and model_cfg
deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

# build task and backend model
task_processor = build_task_processor(model_cfg, deploy_cfg, device)
model = task_processor.build_backend_model(backend_model)

# process input image
input_shape = get_input_shape(deploy_cfg)
model_inputs, _ = task_processor.create_input(image, input_shape)

# do model inference
with torch.no_grad():
    result = model.test_step(model_inputs)

# visualize results
task_processor.visualize(
    image=image,
    model=model,
    result=result[0],
    window_name='visualize',
    output_file='demo/bus_detection.png')