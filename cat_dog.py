import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 数据集预处理
"""
数据来源
知乎https://zhuanlan.zhihu.com/p/388676784

现在目录结构：
root='E:\\kagglecatsanddogs\\'
cat = root + 'PetImages\\Cat'
dog = root + 'PetImages\\Dog'

目标结构：
'data\cad\\train\dog'
'data\cad\\train\cat'

'data\cad\\test\dog'
'data\cad\\test\cat'


"""
def preDealData():

    pass

# 自定义加载数据集
class Dc_DataLoader(Dataset):
    def __init__(self, root, is_train=True):
        super().__init__()
        # 数据集
        self.dataset = []

        train_or_set = 'train' if is_train else 'test'
        # 从路径加载数据


    def __getitem__(self, item):
        pass

    def __len__(self):
        return len(self.dataset)




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
