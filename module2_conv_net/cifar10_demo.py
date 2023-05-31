import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class Net_v1(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
        # nn.ReLU(),
        # nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        # nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1), nn.ReLU(),
        # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1), nn.ReLU(),
        # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        self.conv_layer = nn.Sequential(

            nn.Conv2d(3, 16, 3, 1, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1)
        )
        self.out_layer = nn.Sequential(
            nn.Linear(256 * 2 * 2, 256 * 2), nn.ReLU(),
            nn.Linear(256 * 2, 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        # HCHW转为NV
        x = x.reshape(-1, 256 * 2 * 2)
        return self.out_layer(x)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3, 1)

        self.block1 = self.resnet_block(64, 64)
        self.block2 = self.resnet_block(64, 128)
        self.block3 = self.resnet_block(128, 256)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 10)

    def resnet_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(out_channels, out_channels, 3, 1, padding=1),
            nn.BatchNorm2d(out_channels),

            nn.ReLU()
        )
        return block

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.bn2(x)

        x = self.conv3(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += identity
        x = self.relu(x)
        return x


class ResNet2(nn.Module):
    def __init__(self):
        super(ResNet2, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.residual_block1 = self.residual_block(16, 16)  # 修改这里的参数
        self.residual_block2 = self.residual_block(16, 16)  # 修改这里的参数
        self.residual_block3 = self.residual_block(16, 16)  # 修改这里的参数
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)

    def residual_block(self, in_channels, out_channels):
        block = ResidualBlock(in_channels, out_channels)
        return block

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = self.residual_block3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class Train:
    def __init__(self):
        super().__init__()
        self.train_data = datasets.CIFAR10(
            root=r'data',
            train=True,
            transform=transforms.ToTensor(),
            download=True
        )
        self.test_data = datasets.CIFAR10(
            root=r'data',
            train=False,
            transform=transforms.ToTensor(),
            download=True
        )
        self.train_dataloader = DataLoader(dataset=self.train_data, batch_size=128, shuffle=True)
        self.test_dataloader = DataLoader(dataset=self.test_data, batch_size=256, shuffle=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = ResNet().to(self.device)

        self.optim = torch.optim.Adam(self.net.parameters())
        self.loss = nn.CrossEntropyLoss()

    def trainer(self):
        writer = SummaryWriter('./result')
        train_epoch = 1000
        for epoch in range(1, train_epoch + 1):
            sum_loss = 0.
            for i, (img, target) in enumerate(self.train_dataloader):
                self.net.train()
                target = nn.functional.one_hot(target).float()
                img, target = img.to(self.device), target.to(self.device)

                out = self.net(img)

                loss = self.loss(out, target)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                sum_loss += loss.item()
                # 将训练好的权重文件保存
                torch.save(self.net.state_dict(), f'params3//{i}.pth')

            avg_loss = sum_loss / len(self.train_dataloader)
            # 使用TensorBoard图形化显示
            # 命令行启动board：tensorboard --logdir=result --port=8899
            writer.add_scalar('loss', avg_loss, epoch)
            print('Train Epoch: {} [{}/{} ({:.2f}%)]   Loss: {:.6f}'.format(
                epoch, epoch, train_epoch, 100. * epoch / train_epoch, avg_loss))

            # 每训练十轮进行一次测试
            if epoch % 10 == 0:
                self.test(1)
        print('Train finished!!! ')

    def test(self, e):
        # 载入最优的训练结果，进行测试
        self.net.load_state_dict(torch.load(r'params3//' + os.listdir(r'params3')[-1]))

        for epoch in range(e):
            sum_score = 0
            for i, (img, label) in enumerate(self.test_dataloader):
                # 测试模式
                self.net.eval()
                img, label = img.to(self.device), label.to(self.device)

                # 网络计算答案
                out = self.net(img)
                a = torch.argmax(out, dim=1)
                # 标出答案进行ont-hot
                label = torch.nn.functional.one_hot(label)
                b = torch.argmax(label, dim=1)
                # 计算当前批次的得分
                score = torch.mean(torch.eq(a, b).float())
                sum_score += score
            avg_score = sum_score / len(self.test_dataloader)
            print('Test Epoch: {} [{}/{} ({:.2f}%)]   Score: {:.6f}'.format(
                epoch, epoch, 10, 100. * epoch / 10, avg_score.item()))

        pass


if __name__ == '__main__':
    # 数据格式 nchw 批次，通道数，h，w
    # data = torch.randn(128, 3, 32, 32)

    train = Train()
    # train.trainer()
    train.test(10)
