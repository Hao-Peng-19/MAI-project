import torch
print(torch.__version__)  # 检查 PyTorch 版本
print(torch.cuda.is_available())  # 检查 CUDA 是否可用
print(torch.cuda.get_device_name())  # 获取 GPU 设备名称
