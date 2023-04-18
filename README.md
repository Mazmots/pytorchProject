# 资料地址

d2l官网 http://zh.d2l.ai/
opencv中文文档 https://opencv.apachecn.org/

# 安装pytorch

官网安装，设置conda环境

# 检查是否使用GPU计算

```python
# 检查torch是否正在使用GPU
import torch
a = torch.cuda.is_available()
print(a)

ngpu= 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)
print(torch.cuda.get_device_name(0))
print(torch.rand(3,3).cuda())


```


## Tensors(张量)

view( )用法详解
PyTorch 中的view( )函数相当于numpy中的resize( )函数，都是用来重构(或者调整)张量维度的，用法稍有不同。


view(3, 2)将张量重构成了3x2维的张量。
view(-1)将张量重构成了1维的张量。
torch.view(-1, 参数b)，则表示在参数a未知，参数b已知的情况下自动补齐行向量长度，在这个例子中b=3，re总共含有6个元素，则a=6/2=3。

