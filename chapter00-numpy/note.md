# numpy介绍

NumPy是Python中科学计算的基础包。
它是一个Python库，提供多维数组对象，各种派生对象（如掩码数组和矩阵），以及用于数组快速操作的各种API，有包括数学、逻辑、形状操作、排序、选择、输入输出、离散傅立叶变换、基本线性代数，基本统计运算和随机模拟等等。

NumPy包的核心是 `ndarray` 对象。它封装了python原生的同数据类型的 n 维数组，为了保证其性能优良，其中有许多操作都是代码在本地进行编译后执行的。

NumPy数组 和 原生Python Array（数组）区别：

 - NumPy 数组在创建时具有固定的大小，更改ndarray的大小将创建一个新数组并删除原来的数组。Python原生数组可以动态增长。
 - NumPy 数组中的元素都需要具有相同的数据类型，因此在内存中的大小相同。
 - NumPy 数组有助于对大量数据进行高级数学和其他类型的操作。
 - Numpy底层使用C语言编写，数组中直接存储对象，而不是存储对象指针，所以其运算效率远高于纯
Python代码。

# ndarray与Python原生list运算效率对比
```python
import random
import time
import numpy as np
a = []
for i in range(10000000):
    a.append(random.random())
    
t1 = time.time()
total = sum(a)
t2 = time.time()
print(t2 - t1)

b = np.array(a)
t3 = time.time()
total = np.sum(b)
t4 = time.time()
print(t4 - t3)

```
输出结果:
```python
0.06774353981018066
0.015648841857910156
```

# N维数组-ndarray
## ndarray的属性

 - ndarray.shape: 数组的维度。这是一个整数的元组，表示每个维度中数组的大小。
 - ndarray.ndim: 数组的维度（轴）的个数。
 - ndarray.size: 数组元素的总数。
 - ndarray.itemsize: 数组中每个元素的字节大小。
 - ndarray.dtype: 一个描述数组中元素类型的对象。