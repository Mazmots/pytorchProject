#%% ndarray与Python原生list运算效率对比
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

#%% ndarray属性
import numpy as np

a = np.arange(15).reshape(3, 5)
print(a)
print(a.shape)
print(a.ndim)
print(a.size)
print(a.itemsize)
print(a.dtype)

a.shape = (5, -1)
print(a)


#%%
import numpy as np
print(np.arange(0, 10).reshape(2, -1))
a = np.arange(9).reshape((3, 3))
print(a.itemsize)
print(a.nbytes)


a = np.array([[1,2,3],[4,5,6]],dtype=np.float32)
print(a)
print(a.dtype)

#%% 常用方法

print(np.array([1, 2, 3, 4, 5], ndmin=3))
print(np.nonzero(np.array([1, 0, 2, 3, 0, 4])))