# 9 计算机视觉

## 图像增广常用的方法

### 翻转和裁剪
以给定的概率随机水平翻转给定的图像。
参数：
    p (float)：图像被翻转的概率。 默认值为 0.5
```
torchvision.transforms.RandomHorizontalFlip()
```

以给定的概率随机垂直翻转给定的图像。
```
torchvision.transforms.RandomVerticalFlip()
```
剪图像的随机部分并将其调整为给定大小。

 如果图像是 torch Tensor，则预期 具有 [..., H, W] 形状，其中 ... 表示任意数量的前导尺寸

 对原始图像进行裁剪：裁剪具有随机区域（H * W）
 和随机纵横比。 该裁剪最终调整为给定的大小
 尺寸。 这通常用于训练 Inception 网络。
```
torchvision.transforms.RandomResizedCrop()
```

### 变化颜色
我们可以从4个方面改变图像的颜色：
 - 亮度（brightness）
 - 对比度（contrast）
 - 饱和度（saturation）
 - 色调（hue）。

参数范围 [max(0, 1 - brightness), 1 + brightness]
```
torchvision.transforms.ColorJitter(brightness=0.5)
torchvision.transforms.ColorJitter(hue=0.5)
torchvision.transforms.ColorJitter(contrast=0.5)
torchvision.transforms.ColorJitter(saturation=0.5)
```

### 叠加多个图像增广方法

实际应用中我们会将多个图像增广方法叠加使用。
我们可以通过Compose实例将上面定义的多个图像增广方法叠加起来，再应用到每张图像之上。

```
shape_aug = torchvision.transforms.RandomResizedCrop(
    200, scale=(0.1, 1), ratio=(0.5, 2))

color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), 
    color_aug, 
    shape_aug])
apply(img, augs)

```