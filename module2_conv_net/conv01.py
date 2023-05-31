# %%
import torch.nn as nn
from torchvision import datasets, transforms

# cifar10数据集下载
cifar10_train = datasets.CIFAR10(root=r'data', train=True, transform=transforms.ToTensor(), download=True)

cifar10_test = datasets.CIFAR10(root=r'data', train=False, transform=transforms.ToTensor(), download=True)

print(cifar10_train)
print(cifar10_train.data[0].shape)
print(cifar10_train.targets[0])
print(cifar10_train.classes)
# 图像格式 32*32*3


#

