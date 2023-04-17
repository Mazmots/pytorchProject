"""
核心操作
    图像基本操作
    图像的算术运算
    性能测量和改进技术

"""
"""
 ████ 
▒▒███ 
 ▒███ 
 ▒███ 
 ▒███ 
 ▒███ 
 █████
▒▒▒▒▒ 
图像基本操作
学习：
    访问像素值并修改它们
    访问像素属性
    设置感兴趣区域（ROI）
    拆分和合并图像
"""

#%% 访问和修改像素值
import numpy as np
import cv2 as cv

img = cv.imread('data/images/yellow.png')

# h为高(矩阵行),w为宽(矩阵列),c为通道数
h, w, c = img.shape
print(h,w,c)
print(type(img))
print(img.size)

blue = img[100, 100, 0]
img[1:100,1:100] = [0,0,0]

# cv.imshow('img', img)
# cv.waitKey(0)
# cv.destroyAllWindows()

#%% 感兴趣区域
def showImg(img):
    cv.imshow('img', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

img = cv.imread('data/images/football.png')
b, g, r = cv.split(img)
img[:, :, 1:3] = 0
showImg(img)


#%% 制作图像边界（填充）

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

BLUE = [255, 0, 0]
GREEN = [0, 255, 0]

img1 = cv.imread('data/images/opencv-logo.png')

# 最后一个元素被复制，如下所示： aaaaaa | abcdefgh | hhhhhhh
replicate = cv.copyMakeBorder(img1,50,50,50,50,cv.BORDER_REPLICATE)
# 边框将是边框元素的镜像反射，如下所示：fedcba|abcdefgh|hgfedcb
reflect = cv.copyMakeBorder(img1, 50,50,50,50, cv.BORDER_REFLECT)
# 与上面相同，但略有改动，如下所示： gfedcb | abcdefgh | gfedcba
reflect101 = cv.copyMakeBorder(img1,50,50,50,50,cv.BORDER_REFLECT_101)
# 它看起来像这样： cdefgh | abcdefgh | abcdefg
wrap = cv.copyMakeBorder(img1,50,50,50,50,cv.BORDER_WRAP)
# 添加一个恒定的彩色边框。该值应作为下一个参数value给出。
# 由于plt中和cv中的B和R值轴相反，所以绘图会出相反值
constant= cv.copyMakeBorder(img1,50,50,50,50,cv.BORDER_CONSTANT,value=BLUE)


plt.subplot(231), plt.imshow(img1, 'gray'), plt.title('ORIGINAL')
plt.subplot(232), plt.imshow(replicate,'gray'), plt.title('REPLICATE')
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')
plt.show()

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

图像算术运算
    学习对图像的几种算术运算，如加法，减法，按位运算等。
"""

#%% 图像混合 q(x) = (1-α)f0(x) + αf1(f)
img1 = cv.imread('data/images/1.jpg')
img2 = cv.imread('data/images/opencv-logo11.png')


dst = cv.addWeighted(img1, 0.1, img2, 0.9, 0)
cv.imshow('dst', dst)
cv.waitKey(0)
cv.destroyAllWindows()


#%% 按位操作
#加载两张图片
img1 = cv.imread('data/images/1.jpg')
img2 = cv.imread('data/images/opencv-logo.png')
#我想在左上角放置一个logo，所以我创建了一个 ROI,并且这个ROI的宽和高为我想放置的logo的宽和高
rows,cols,channels = img2.shape
roi = img1 [0:rows,0:cols]
#现在创建一个logo的掩码，通过对logo图像进行阈值，并对阈值结果并创建其反转掩码
img2gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

ret,mask = cv.threshold(img2gray,150,255,cv.THRESH_BINARY)


mask_inv = cv.bitwise_not(mask)

# showImg(mask)
#现在使 ROI 中的徽标区域变黑
img1_bg = cv.bitwise_and(roi,roi,mask = mask)

showImg(img1_bg)
#仅从徽标图像中获取徽标区域。
img2_fg = cv.bitwise_and(img2,img2,mask = mask_inv)
#在 ROI 中放置徽标并修改主图像
dst = cv.add(img1_bg,img2_fg)
img1 [0:rows,0:cols] = dst
cv.imshow('res',img1)
cv.waitKey(0)
cv.destroyAllWindows()



