#%% 导包
import time
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image

import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 读取一张
d2l.set_figsize()
img = Image.open(r'D:\PycharmProjects\pytorchProject\chapter99-pytorch\c09-computerView\img\1.jpg')
d2l.plt.imshow(img)
d2l.plt.show()


#%% 绘图函数
# 本函数已保存在d2lzh_pytorch包中方便以后使用
def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    d2l.plt.show()
    return axes

# 辅助函数apply。这个函数对输入图像img多次运行图像增广方法aug并展示所有的结果。
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_images(Y, num_rows, num_cols, scale)


#%%
# 用于对载入的图片按随机概率进行水平翻转。
# 我们可以通过传递给这个类的参数自定义随机概率，如果没有定义，则使用默认的概率值0.5。
apply(img, torchvision.transforms.RandomHorizontalFlip())

#%%
# 垂直翻转
apply(img, torchvision.transforms.RandomVerticalFlip())
#%%
# 可以通过对图像随机裁剪来让物体以不同的比例出现在图像的不同位置，这同样能够降低模型对目标位置的敏感性。
# 在下面的代码里，我们每次随机裁剪出一块面积为原面积10%∼100%的区域，且该区域的宽和高之比随机取自0.5∼2
# 然后再将该区域的宽和高分别缩放到200像素。若无特殊说明，本节中a和b之间的随机数指的是从区间[a,b]中随机均匀采样所得到的连续值。
shape_aug = torchvision.transforms.RandomResizedCrop(200, scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)


#%% 颜色变化
# 我们可以从4个方面改变图像的颜色：
# 亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue）。

# 我们将图像的亮度随机变化为原图亮度的50%50%（1−0.51−0.5）∼150%∼150%（1+0.51+0.5）。
apply(img, torchvision.transforms.ColorJitter(brightness=0.5))

#%%
# 对比度
apply(img, torchvision.transforms.ColorJitter(contrast=0.5))

#%%
# 饱和度
apply(img, torchvision.transforms.ColorJitter(saturation=0.5))

#%%
# 色调
apply(img, torchvision.transforms.ColorJitter(hue=0.5))

#%%
# 同时变换
color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)

#%% 多方法叠加
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    shape_aug,
    color_aug
])
apply(img, augs)


############################################################################

#%%
# 使用图像增广训练模型
all_imges = torchvision.datasets.CIFAR10(train=True, root="~/Datasets/CIFAR", download=True)
# all_imges的每一个元素都是(image, label)
show_images([all_imges[i][0] for i in range(32)], 4, 8, scale=0.8);

#%%
# 为了在预测时得到确定的结果，我们通常只将图像增广应用在训练样本上，而不在预测时使用含随机操作的图像增广。
# 在这里我们只使用最简单的随机左右翻转。
# 此外，我们使用ToTensor将小批量图像转成PyTorch需要的格式，即形状为(批量大小, 通道数, 高, 宽)、值域在0到1之间且类型为32位浮点数。

flip_aug = torchvision.transforms.Compose([
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor()])

no_aug = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor()])

#%%
# 接下来我们定义一个辅助函数来方便读取图像并应用图像增广。
num_workers = 0 if sys.platform.startswith('win32') else 4
def load_cifar10(is_train, augs, batch_size, root="~/Datasets/CIFAR"):
    dataset = torchvision.datasets.CIFAR10(root=root, train=is_train, transform=augs, download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers)

# 使用图像增广训练模型
# 我们先定义train函数使用GPU训练并评价模型。
# 本函数已保存在d2lzh_pytorch包中方便以后使用
def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = d2l.evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

# 然后就可以定义train_with_data_aug函数使用图像增广来训练模型了。
# 该函数使用Adam算法作为训练使用的优化算法，然后将图像增广应用于训练数据集之上，最后调用刚才定义的train函数训练并评价模型。
def train_with_data_aug(train_augs, test_augs, lr=0.001):
    batch_size, net = 256, d2l.resnet18(10)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss()
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    train(train_iter, test_iter, net, loss, optimizer, device, num_epochs=10)

# 下面使用随机左右翻转的图像增广来训练模型。
train_with_data_aug(flip_aug, no_aug)

