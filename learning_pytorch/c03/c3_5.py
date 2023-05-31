#%%图像分类数据集¶
import os

import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()

# 读取数据集
trans = transforms.ToTensor()

mnist_train = torchvision.datasets.FashionMNIST(
    root='data', train=True, transform=trans, download=True)

mnist_test = torchvision.datasets.FashionMNIST(
    root='data', train=False, transform=trans, download=True)

print(len((mnist_train)))
print(len(mnist_test))

#%%
def get_fashion_mnist_labels(labels):  #@save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))


data.DataLoader

#%%
import torch
from torch.utils.data import Dataset,tra
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = os.listdir(root)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.images[index])
        image = Image.open(r'E:\pyprojects\pytorchProject\pytorch_learn\c03\b.png')
        if self.transform is not None:
            image = self.transform(image)
        return image

root = 'images'
transform = transforms.Compose([
    transforms.Resize(256),  # 将图像resize到256x256
    transforms.CenterCrop(224), # 从中心裁剪出224x224的图像
    transforms.ToTensor()  # 将Image转为Tensor
])
dataset = ImageDataset(root, transform)

img = dataset[0]  # 读取第一张图像
print(img.size()) # torch.Size([3, 224, 224])