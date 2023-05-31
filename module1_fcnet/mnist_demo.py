#%%

import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets.mnist as mnist
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# 定义数据集
"""
# 训练集
train_dataset = datasets.MNIST(root='data/', train=True, transform=transforms.ToTensor(), download=True)
# 测试集
test_dataset = datasets.MNIST(root='data/', train=False, transform=transforms.ToTensor(), download=True)

print(train_dataset.data.shape)
print(test_dataset.data.shape)

a = train_dataset.data[0]
print(a)
img = transforms.ToPILImage()(a)
img.show()
"""
root = r'E:\pyprojects\pytorchProject\data\MNIST\raw'


train_set = mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),\
    mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))

test_set = mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),\
    mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))

# 训练集
train_path = os.path.join(root, 'train')
if not os.path.exists(train_path):
    os.makedirs(train_path)

# 迭代：数据，标签
# for i, (img, label) in enumerate(zip(train_set[0], train_set[1])):
#
#     # 构建路径
#     path = f'{train_path}//{label}//{i}.jpg'
#     if not os.path.exists(f'{train_path}//{label}'):
#         os.makedirs(f'{train_path}//{label}')
#     # 写文件操作
#     io.imsave(path, np.array(img))

print(f'训练集数据写入完成')


# 测试集
# test_path = os.path.join(root, 'test')
# if not os.path.exists(test_path):
#     os.makedirs(test_path)
#
# for i, (img, label) in enumerate(zip(test_set[0], test_set[1])):
#     # 构建路径
#     path = f'{test_path}//{label}//{i}.jpg'
#     if not os.path.exists(f'{test_path}//{label}'):
#         os.makedirs(f'{test_path}//{label}')
#
#     # 写入文件
#     io.imsave(path, np.array(img))

print(f'测试集数据写入完成')


# 构建网络
class MNIST_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(1 * 28 * 28, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 10),
            # 神经网络输入是NV结构，
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fc_layer(x)

# data = torch.randn(128, 1* 28 * 28)
# print(MNIST_Net()(data).shape)

# 封装数据集
class MNIST_Dataset(Dataset):
    def __init__(self, root, is_train=True):
        super().__init__()
        # 数据集
        self.dataset = []

        train_or_test = 'train' if is_train else 'test'
        # 从路径加载数据
        path = f'{root}//{train_or_test}'
        for label in os.listdir(path):

            for img_path in os.listdir(f'{path}//{label}'):
                file = f'{path}//{label}//{img_path}'
                self.dataset.append((file, label))
        pass

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]

        # 使用opencv读取图像
        img = cv2.imread(data[0], 0)
        img.reshape(-1)
        # 归一化
        img = img / 255
        # 标签one hot
        one_hot = np.zeros(10)
        one_hot[int(data[1])] = 1

        return np.float32(img), np.float32(one_hot)


# dataset = MNIST_Dataset(root, False)

# 训练器
class Trainer:
    def __init__(self):
        super().__init__()

        # 判断是否有cuda
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 网络
        self.net = MNIST_Net()
        # net放入cuda 或cpu
        self.net.to(self.device)

        self.train_dataset = MNIST_Dataset(root=r'E:\pyprojects\pytorchProject\data\MNIST\raw', is_train=True)
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=2048, shuffle=True)

        self.test_dataset = MNIST_Dataset(root=r'E:\pyprojects\pytorchProject\data\MNIST\raw', is_train=False)
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=256, shuffle=True)

        # 模型训练完成，得到h，loss，反向更新，优化器优化模型的权重

        # self.opt = torch.optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=0.001)


    def train(self):
        writer = SummaryWriter('./result')
        print('==============训练开始==============')
        for epoch in range(1000):
            sum_loss = 0
            for i, (img, label) in enumerate(self.train_loader):
                # 模型操作，打开训练模式
                self.net.train()
                img, label = img.to(self.device), label.to(self.device)
                img = img.reshape(-1, 784)
                # 前向计算
                h = self.net(img)
                # 求损失
                loss = torch.mean((h - label) ** 2)
                # 梯度清空
                self.opt.zero_grad()
                # 反向更新  更新w和b
                loss.backward()
                # 计算梯度
                self.opt.step()

                sum_loss += loss

                # 写入权重文件
                torch.save(self.net.state_dict(), f'params//{i}.pth')


            avg_loss = sum_loss / len(self.train_loader)

            # 使用TensorBoard图形化显示
            # 命令行启动board：tensorboard --logdir=result --port=8899
            writer.add_scalar('loss', avg_loss, epoch)

            print(f'第{epoch}轮次的损失是{avg_loss.item()}')
        pass

    def test(self):
        # 把最优的训练效果进行测试
        self.net.load_state_dict(torch.load(r'params//' + os.listdir(r'params')[-1]))
        print('==============测试开始==============')

        for epoch in range(10000):
            sum_score = 0
            for i, (img, label) in enumerate(self.test_loader):
                # 测试模式
                self.net.eval()
                img, label = img.to(self.device), label.to(self.device)
                img = img.reshape(-1, 784)

                h = self.net(img)

                # h为网络计算答案,label标准答案
                a = torch.argmax(h, dim=1)
                b = torch.argmax(label, dim=1)
                # 当前批次得分
                score = torch.mean(torch.eq(a, b).float())
                sum_score += score
            avg_score = sum_score / len(self.test_loader)
            print(f'第{epoch}轮的得分是{avg_score}')
        pass

if __name__ == '__main__':

    trainer = Trainer()
    trainer.train()

    # trainer.test()