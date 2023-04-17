import torch

#%%
# 构造5*3的矩阵
x = torch.empty(5,3)
print(x)

#%%
# 随机矩阵
x = torch.rand(5, 3)
print(x)

#%%
# 0矩阵
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

#%%
# 构造一个张量，直接使用数据：
x = torch.tensor([5.5, 3])
print(x)

#%%
# 加法操作
x = torch.rand(5, 3)
y = torch.rand(5, 3)
print(torch.add(x, y))

#%%
# 张量大小改变
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)

print(x)
print(y)
print(z)
print(x.size(), y.size(), z.size())

#%%
# 以下代码只有在PyTorch GPU版本上才会执行
if torch.cuda.is_available():
    device = torch.device("cuda")          # GPU
    y = torch.ones_like(x, device=device)  # 直接创建一个在GPU上的Tensor
    x = x.to(device)                       # 等价于 .to("cuda")
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # to()还可以同时更改数据类型

#%% matplotlib inline

import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs,
                       dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                       dtype=torch.float32)
print(features[0], labels[0])

#%%
def use_svg_display():
    # 用矢量图显示
    display.display_svg()

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize

# # 在../d2lzh_pytorch里面添加上面两个函数后就可以这样导入
# import sys
# sys.path.append("..")
# from d2lzh_pytorch import *

set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1);
plt.show()