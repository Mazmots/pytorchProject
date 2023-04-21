#%%
import time

import torch
import torch.nn as nn

# print(torch.cuda.is_available())
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

class Net01(nn.Module):
    def __init__(self):
        super().__init__()

        # 神经网络序列构造器
        self.layer = nn.Sequential(
            # 构造每一层

            # in_features: 输入的神经元个数
            # out_features: 输出神经元个数
            # bias：默认True  是否包含偏置
            nn.Linear(in_features=3 * 450 * 280, out_features=1024, bias=True),
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 2),

            # 分类，输出函数，NV结构
            nn.Softmax(dim=1)
        )

    # 前向计算
    # x为输入网络的数据
    def forward(self, x):
        return self.layer(x)

start = time.time()
# net = Net().to(device)
net = Net01()
print(net)

# data = torch.randn(128, 3 * 450 * 280).to(device)
data = torch.randn(128, 3 * 450 * 280)
h = net.forward(data)
print(h.shape)
print(h)

print(f'{time.time()-start}')