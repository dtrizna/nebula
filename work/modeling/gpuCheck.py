import os
import torch

print(os.system("dkms status")) # nvidia driver

print(os.system("lshw -C display"))

print(os.system("lspci | grep -i nvidia"))

print(os.system("nvcc --version"))

print(torch.cuda.is_available())

print(torch.cuda.current_device())

print(torch.cuda.device(0))

print(torch.cuda.device_count())

print(torch.cuda.get_device_name(0))

print(torch.__version__)

print(torch.tensor([1.0, 2.0]).cuda()) # actual check if can use cuda