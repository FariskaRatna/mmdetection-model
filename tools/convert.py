from mmdeploy.apis import torch2onnx
# from mmdeploy.backend.sdk.export_info import export2SDK

img = 'data/val_bus2024/000000005037.jpg'
work_dir = 'work_dirs/mmdeploy_models/mmdet/onnx'
save_file = 'bus_onnx12.onnx'
deploy_cfg = 'tools/onnxruntime_dynamic.py'
model_cfg = 'configs/faster_rcnn/faster-rcnn_r101_fpn_1x_bus.py'
model_checkpoint = 'work_dirs/faster-rcnn_r101_fpn_1x_bus/epoch_10.pth'
device = 'cpu'

# 1. convert model to onnx
torch2onnx(img, work_dir, save_file, deploy_cfg, model_cfg,
           model_checkpoint, device)

# 2. extract pipeline info for inference by MMDeploy SDK
# export2SDK(deploy_cfg, model_cfg, work_dir, pth=model_checkpoint,
#            device=device)