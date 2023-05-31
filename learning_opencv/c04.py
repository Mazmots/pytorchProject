"""
图像处理
    更改颜色空间
    图像的几何变换
    图像阈值
    平滑图像
    形态转换
    图像梯度
    Canny边缘检测
    图像金字塔
    OpenCV中的轮廓
    OpenCV中的直方图
    OpenCV中的图像转换
    模板匹配
    霍夫线变换
    霍夫圆变换
    基于分水岭算法的图像分割
    基于GrabCut算法的交互式前景提取

"""
import cv2
import numpy as np

"""
 ████ 
▒▒███ 
 ▒███ 
 ▒███ 
 ▒███ 
 ▒███ 
 █████
▒▒▒▒▒ 
更改颜色空间
"""
#%%
"""
cv.cvtColor(input_image, flag)
两个最常使用的方法，
    BGR 到 Gray: cv.COLOR_BGR2GRAY
    BGR 到 HSV: cv.COLOR_BGR2HSV
"""
import cv2 as cv
# flags = [i for i in dir(cv) if i.startswith('COLOR_')]
# print( flags )

img = cv.imread('data/images/1.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV_FULL)

cv.imshow('img gray', img_gray)
cv.imshow('img hsv', img_hsv)
cv.waitKey(0)
cv.destroyAllWindows()

#%% 目标追踪
import cv2 as cv
import numpy as np
cap = cv.VideoCapture(0)

# 通过rbg值追踪hsv值
def get_mask(color):
    color_rgb = np.uint8([[[0, 0, 0]]])
    if color == 'blue':
        color_rgb = np.uint8([[[0, 0, 255]]])
    elif color == 'green':
        color_rgb = np.uint8([[[0, 255, 0]]])

    color_hsv = cv.cvtColor(color_rgb, cv.COLOR_RGB2HSV).reshape(-1)
    blue_mask = cv.inRange(hsv,
                           np.asarray([color_hsv[0] - 30, 100, 100]),
                           np.asarray([color_hsv[0] + 30, 255, 255]))
    return blue_mask

while(True):
    _, frame = cap.read()
    # rgb转hsv
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # define range of blue color in HSV
    # lower_blue = np.array([100,100,100])
    # upper_blue = np.array([130,255,255])
    # Threshold the HSV image to get only blue colors
    # blue_mask = cv.inRange(hsv, lower_blue, upper_blue)
    blue_mask = get_mask('blue')
    green_mask = get_mask('green')
    # Bitwise-AND mask and original image
    blue_res = cv.bitwise_and(frame, frame, mask=blue_mask)
    green_res = cv.bitwise_and(frame, frame, mask=green_mask)

    cv.imshow('frame', frame)
    cv.imshow('blue mask', blue_mask)
    cv.imshow('blue res', blue_res)
    cv.imshow('green mask', green_mask)
    cv.imshow('green res', green_res)

    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()


#%%
"""
  ████████ 
 ███▒▒▒▒███
▒▒▒    ▒███
   ███████ 
  ███▒▒▒▒  
 ███      █
▒██████████
▒▒▒▒▒▒▒▒▒▒ 

图像的几何变换
    如平移、旋转、仿射变换等。
"""

#%%
# 两个转换函数，
# cv.warpAffine 和 cv.warpPerspective，可以进行各种转换

import numpy as np
import cv2 as cv

img = cv.imread('data/images/1.jpg')
# interpolation :插值方法
# 对于下采样(图像上缩小)，最合适的插值方法是 cv.INTER_AREA
# 对于上采样(放大),最好的方法是 cv.INTER_CUBIC （速度较慢）和 cv.INTER_LINEAR (速度较快)。
# 默认情况下，所使用的插值方法都是 cv.INTER_AREA 。
# 下采样
res1 = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
res2 = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)

cv.imshow('res1', res1)
cv.imshow('res2', res2)
cv.waitKey(0)
cv.destroyAllWindows()


#%% 平移变换
img = cv.imread('data/images/1.jpg')
rows,cols, _ = img.shape

"""
平移变换是物体位置的移动。如果知道 （x，y） 方向的偏移量，
假设为 (t_x,t_y)**，则可以创建如下转换矩阵 **M：
    
    M = [ 1 0 tx ]
        [ 0 1 ty ]
"""
M = np.float32([[1,0,100], [0,1,50]])
# cv.warpAffine 函数的第三个参数是输出图像的大小，其形式应为（宽度、高度）。
# 记住宽度=列数，高度=行数。
dst = cv.warpAffine(img, M, (cols, rows))

cv.imshow('dst', dst)
cv.waitKey(0)
cv.destroyAllWindows()

#%% 旋转
img = cv.imread('data/images/1.jpg')

rows,cols,_ = img.shape
# cols-1 and rows-1 are the coordinate limits.
M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),45,1)
dst = cv.warpAffine(img,M,(cols,rows))

cv.imshow('dst', dst)
cv.waitKey(0)
cv.destroyAllWindows()

#%%
"""
  ████████ 
 ███▒▒▒▒███
▒▒▒    ▒███
   ██████▒ 
  ▒▒▒▒▒▒███
 ███   ▒███
▒▒████████ 
 ▒▒▒▒▒▒▒▒  
图像阈值
"""
#%% 简单阈值法
"""
如果像素值大于阈值，则会被赋为一个值（可能为白色），否则会赋为另一个值（可能为黑色）。
使用的函数是 cv.threshold。
    第一个参数是源图像，它应该是灰度图像。
    第二个参数是阈值，用于对像素值进行分类。
    第三个参数是 maxval，它表示像素值大于（有时小于）阈值时要给定的值。
    opencv 提供了不同类型的阈值，由函数的第四个参数决定。类型有：
        cv.THRESH_BINARY  
            二值化阈值处理：在对8位灰度图像进行二值化时，如果将阈值设定为127，那么：
                         所有大于127的像素点会被处理为255。其余值会被处理为0。
        cv.THRESH_BINARY_INV  
            反二值化阈值处理：对于灰度值大于阈值的像素点，将其值设定为0。
                         对于灰度值小于或等于阈值的像素点，将其值设定为255。
        cv.THRESH_TRUNC  
            截断阈值化处理：将图像中大于阈值的像素点的值设定为阈值，小于或等于该阈值的像素点的值保持不变
        cv.THRESH_TOZERO  
            低阈值零处理：将图像中小于或等于阈值的像素点的值处理为0，大于阈值的像素点的值保持不变。
        cv.THRESH_TOZERO_INV
            超阈值零处理：超阈值零处理会将图像中大于阈值的像素点的值处理为0，小于或等于该阈值的像素点的值保持不变。
"""
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('data/images/1.jpg', 0)
_, thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
_, thresh2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
_, thresh3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
_, thresh4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
_, thresh5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)

titles = ['Original', 'BINARY', 'BINARY_INV',
          'TRUNC', 'TOZERO', 'TOZERO_INV']

images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2,3,i+1),
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])

plt.show()

#%% 自适应阈值
"""
自适应阈值算法计算图像的一个小区域的阈值。
因此，我们得到了同一图像不同区域的不同阈值，对于不同光照下的图像，得到了更好的结果。

"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('data/images/digital.jpg', 0)
# 使用中值滤波器模糊图像。
# 该函数使用中值滤波器平滑图像ksize × ksize光圈。
# 多通道图像的每个通道都是独立处理的。支持就地操作。
# img = cv.medianBlur(img, 5)

ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                           cv.THRESH_BINARY,11,2)
th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                           cv.THRESH_BINARY,11,2)
titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']

images = [img, th1, th2, th3]

for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()


#%%
"""
 █████ █████ 
▒▒███ ▒▒███  
 ▒███  ▒███ █
 ▒███████████
 ▒▒▒▒▒▒▒███▒█
       ▒███▒ 
       █████ 
      ▒▒▒▒▒  
平滑图像
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('data/images/opencv-logo.png')
kernel = np.ones((5,5),np.float32)/25
print(img)
print(kernel)

dst = cv.filter2D(img,-1,kernel)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()


#%%
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('data/images/digital.jpg',0)
laplacian = cv.Laplacian(img,cv.CV_64F)
sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)
plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()


#%%
"""
 ██████████
▒███▒▒▒▒▒▒█
▒███     ▒ 
▒█████████ 
▒▒▒▒▒▒▒▒███
 ███   ▒███
▒▒████████ 
 ▒▒▒▒▒▒▒▒  
形态转换
"""

#%% 腐蚀 膨胀
# 假设前景色为白色，通过卷积，可以消除白色噪音
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread(r'E:\cao\pytorchProject\data\images\i.png')

# 有当内核下的所有像素都为 1 时，原始图像中的像素（1 或 0）才会被视为 1，否则会被侵蚀（变为零）。
blur = cv.blur(img, (4,4))
# 这里，如果内核下至少有一个像素为“1”，则像素元素为“1”。
# 所以它会增加图像中的白色区域，或者增加前景对象的大小。
dilation = cv.dilate(blur,(3,3),iterations = 1)

plt.subplot(131), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])

plt.subplot(132), plt.imshow(blur), plt.title('blur')
plt.xticks([]), plt.yticks([])

plt.subplot(133), plt.imshow(dilation), plt.title('dilation')
plt.xticks([]), plt.yticks([])

plt.show()


#%% 开运算（腐蚀）  闭运算（膨胀后腐蚀）
# 开运算用于去除噪点
# 闭运算用于填充前景上的小孔


import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread(r'E:\cao\pytorchProject\data\images\i.png')
kernel = (15,15)
opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
# 形态梯度 它是图像的膨胀和腐蚀之间的差值。 结果将类似于对象的轮廓。
gradient = cv.morphologyEx(closing, cv.MORPH_GRADIENT, kernel)
# 顶帽 原图像和原图像开运算结果的差值
tophat = cv.morphologyEx(closing, cv.MORPH_TOPHAT, kernel)
# 7、黑帽 它是原图像和原图像的闭的差值
blackhat = cv.morphologyEx(closing, cv.MORPH_BLACKHAT, kernel)

plt.subplot(231), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])

plt.subplot(232), plt.imshow(opening), plt.title('opening')
plt.xticks([]), plt.yticks([])

plt.subplot(233), plt.imshow(closing), plt.title('closing')
plt.xticks([]), plt.yticks([])

plt.subplot(234), plt.imshow(gradient), plt.title('gradient')
plt.xticks([]), plt.yticks([])

plt.subplot(235), plt.imshow(tophat), plt.title('tophat')
plt.xticks([]), plt.yticks([])

plt.subplot(236), plt.imshow(blackhat), plt.title('blackhat')
plt.xticks([]), plt.yticks([])

plt.show()

"""
  ████████ 
 ███▒▒▒▒███
▒███   ▒▒▒ 
▒█████████ 
▒███▒▒▒▒███
▒███   ▒███
▒▒████████ 
 ▒▒▒▒▒▒▒▒  
图像梯度
"""

#%%
# OpenCv 提供三种类型的梯度滤波器或高通滤波器，Sobel、Scharr 和 Laplacian。
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('data/images/digital.jpg', 0)
laplacian = cv.Laplacian(img, cv.CV_64F)

sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)

plt.subplot(221), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.xticks([]), plt.yticks([])

plt.subplot(222), plt.imshow(laplacian, cmap='gray'), plt.title('Laplacian')
plt.xticks([]), plt.yticks([])

plt.subplot(223), plt.imshow(sobelx, cmap='gray'), plt.title('Sobel X')
plt.xticks([]), plt.yticks([])

plt.subplot(224), plt.imshow(sobely, cmap='gray'), plt.title('Sobel Y')
plt.xticks([]), plt.yticks([])
plt.show()

"""
 ██████████
▒███▒▒▒▒███
▒▒▒    ███ 
      ███  
     ███   
    ███    
   ███     
  ▒▒▒      
Canny边缘检测
"""


#%%
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# img = cv.imread('data/images/canny.png',0)
# 有噪声
img = cv.imread('data/images/canny2.png',0)
# 降噪

gaussian_blur = cv.GaussianBlur(img, (7, 7), 0)

edges = cv.Canny(gaussian_blur,100,200)
plt.subplot(131),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(gaussian_blur,cmap = 'gray')
plt.title('gaussian_blur'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()


"""
  ████████  
 ███▒▒▒▒███ 
▒███   ▒███ 
▒▒████████  
 ███▒▒▒▒███ 
▒███   ▒███ 
▒▒████████  
 ▒▒▒▒▒▒▒▒   

"""

