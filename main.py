# %%
import os

import cv2
import torch.cuda
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn

# 训练集
train_dataset = datasets.MNIST(root='datafiles/', train=True, transform=transforms.ToTensor(), download=True)
# 测试集
test_dataset = datasets.MNIST(root='datafiles/', train=False, transform=transforms.ToTensor(), download=True)


# 自定义全连接网络
class FC_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_layer = nn.Sequential(
            # 图像为1*28*28，因此输入参数为784
            nn.Linear(784, 512),
            # 激活函数使用relu
            nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 10),
            # 最后使用Softmax激活
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fc_layer(x)


class Trainer:
    def __init__(self):
        # 定义训练参数
        # 学习率
        self.lr = 0.001
        # 训练集批次
        self.train_batch_size = 512
        # 测试集批次
        self.test_batch_size = 256
        # 训练迭代次数
        self.train_epoch = 10000
        # 测试迭代次数
        self.test_epoch = 1000

        # 判断是否有cuda，如果有，将net放到gpu进行训练
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 网络模型
        self.net = FC_Net()
        # 将网络放入cuda或cpu
        self.net.to(self.device)

        # 装载训练集和测试集
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.train_batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=self.test_batch_size, shuffle=True)

        # 定义优化器
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        pass

    def train(self):
        train_epoch = self.train_epoch
        for epoch in range(train_epoch):
            # 总损失
            sum_loss = 0
            for i, (data, label) in enumerate(self.train_loader):
                # 模型训练，打开训练模式
                self.net.train()
                # 数据放入cuda
                data, label = data.to(self.device), label.to(self.device)
                data = data.reshape(-1, 784)
                # print(data.shape)
                # label 进行one-hot处理
                label = torch.nn.functional.one_hot(label)
                self.net.train()
                # 前向计算
                h = self.net(data)
                # 求损失 使用均方差公式
                loss = torch.mean((h - label) ** 2)
                # 清空过往梯度
                self.opt.zero_grad()
                # 反向传播，计算当前梯度
                loss.backward()
                # 根据梯度更新网络参数
                self.opt.step()

                sum_loss += loss
                # 将训练好的权重文件保存
                torch.save(self.net.state_dict(), f'params//{i}.pth')

            avg_loss = sum_loss / len(self.train_loader)

            print('Train Epoch: {} [{}/{} ({:.2f}%)]   Loss: {:.6f}'.format(
                epoch, epoch, self.train_epoch, 100. * epoch / self.train_epoch, avg_loss.item()))

    def test(self):
        # 载入最优的训练结果，进行测试
        self.net.load_state_dict(torch.load(r'params//' + os.listdir(r'params')[-1]))

        for epoch in range(self.test_epoch):
            sum_score = 0
            for i, (img, label) in enumerate(self.test_loader):
                # 测试模式
                self.net.eval()
                img, label = img.to(self.device), label.to(self.device)
                img = img.reshape(-1, 784)

                # 网络计算答案
                h = self.net(img)
                a = torch.argmax(h, dim=1)
                # 标准答案
                label = torch.nn.functional.one_hot(label)
                b = torch.argmax(label, dim=1)

                # 计算当前批次的得分
                score = torch.mean(torch.eq(a, b).float())
                sum_score += score
            avg_score = sum_score / len(self.test_loader)
            print('Test Epoch: {} [{}/{} ({:.2f}%)]   Score: {:.6f}'.format(
                epoch, epoch, self.test_epoch, 100. * epoch / self.test_epoch, avg_score.item()))

        pass


def mytest():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 网络模型
    net = FC_Net()
    # 将网络放入cuda或cpu

    net.load_state_dict(torch.load(r'params//' + os.listdir(r'params')[-1]))

    img = cv2.imread('data/1.png', 0)
    net.eval()
    h = net(torch.Tensor(img.reshape(-1, 784)))
    print(f'预测的数字是：{torch.argmax(h)}')


if __name__ == '__main__':
    trainer = Trainer()
# trainer.train()
# trainer.test()
mytest()
