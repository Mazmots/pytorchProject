# %% 检查torch是否正在使用GPU
import torch

a = torch.cuda.is_available()
print(torch.__version__)
print(torch.version.cuda)
print(a)

ngpu = 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)
print(torch.cuda.get_device_name(0))
print(torch.rand(3, 3).cuda())

"""
cuda配置方法：
    安装cuda，安装cudnn    检查是否安装成功可参考https://blog.csdn.net/m511655654/article/details/88419965
    安装pytorch
    
    anaconda安装   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
"""
