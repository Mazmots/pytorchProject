import gc
import os.path
import shutil

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from net.Res_net import Res_net
from net.Fc_net import Fc_net
from net.Conv_net import Conv_net

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class CatDogDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = self.get_image_paths()

    def get_image_paths(self):
        image_paths = []
        for subdir in os.listdir(self.root_dir):
            subdir_path = os.path.join(self.root_dir, subdir)
            if os.path.isdir(subdir_path):
                for filename in os.listdir(subdir_path):
                    if filename.endswith('.jpg'):
                        image_path = os.path.join(subdir_path, filename)
                        image_paths.append(image_path)
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        label = 0 if 'cat' in image_path else 1  # 根据文件路径确定标签，猫为0，狗为1
        one_hot = F.one_hot(torch.tensor(label), num_classes=2).float()  # 进行独热编码

        return np.float32(image), np.float32(one_hot)


# 自定义训练器
class Trainer:
    def __init__(self):
        # 数据集路径
        data_dir = r'E:/catdog/train'

        transform = transforms.Compose([transforms.ToTensor()])
        # 创建自定义数据集实例
        dataset = CatDogDataset(data_dir, transform=transform)
        # 创建数据加载器
        self.train_loader = DataLoader(dataset, batch_size=2, shuffle=True, pin_memory=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.net = Conv_net()
        self.net.to(self.device)

        self.optim = torch.optim.Adam(self.net.parameters(), lr=0.001)

    def train(self):
        train_epoch_size = 500
        for epoch in range(train_epoch_size):
            sum_loss = 0
            for i, (img, label) in enumerate(self.train_loader):
                self.net.train()
                img, label = img.to(self.device), label.to(self.device)
                h = self.net(img)

                loss = torch.mean((h - label) ** 2)
                self.optim.zero_grad()

                self.optim.step()

                sum_loss += loss

            avg_loss = sum_loss / len(self.train_loader)
            print(f'第{epoch}轮次的损失是{avg_loss.item()}')
            gc.collect()
            torch.cuda.empty_cache()

    def test(self):
        pass


if __name__ == '__main__':
    # deal_data()
    trainer = Trainer()
    trainer.train()
