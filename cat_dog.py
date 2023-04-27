import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


"""
知乎https://zhuanlan.zhihu.com/p/388676784
"""
# 自定义加载数据集
class Dc_DataLoader(Dataset):
    def __init__(self):
        super().__init__()

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass




# 自定义模型
class Dc_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_net = nn.Sequential(
            nn.Linear(), nn.ReLU(),
            nn.Linear(), nn.ReLU(),
            nn.Linear(),
            nn.Softmax()
        )

    def backward(self, x):
        return self.fc_net(x)


# 自定义训练器
class Trainer:
    def __init__(self):
        pass

    def train(self):
        pass
    def test(self):
        pass

# if __name__ == '__main__':
