import torch

#%%
# 构造5*3的矩阵
x = torch.empty(5,3)
print(x)

#%%
# 随机矩阵
x = torch.rand(5, 3)
print(x)

#%%
# 0矩阵
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

#%%
# 构造一个张量，直接使用数据：
x = torch.tensor([5.5, 3])
print(x)

#%%
# 加法操作
x = torch.rand(5, 3)
y = torch.rand(5, 3)
print(torch.add(x, y))

#%%
# 张量大小改变
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)

print(x)
print(y)
print(z)
print(x.size(), y.size(), z.size())

