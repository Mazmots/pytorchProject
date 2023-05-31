"""
图像直方图  显示图像的强度分布

 cv.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
    images：它是uint8或float32类型的源图像。它应该放在方括号中，即“ [img]”。
    channels：也以方括号给出。它是我们计算直方图的通道的索引。例如，如果输入为灰度图像，则其值为[0]。对于彩色图像，您可以传递[0]，[1]或[2]分别计算蓝色，绿色或红色通道的直方图。
    mask：遮罩图像。为了找到完整图像的直方图，将其指定为“无”。但是，如果要查找图像特定区域的直方图，则必须为此创建一个遮罩图像并将其作为遮罩。(我将在后面显示一个示例。)
    histSize：这表示我们的BIN计数。需要放在方括号中。对于全尺寸，我们通过[256]。
    ranges：这是我们的RANGE。通常为[0,256]。 因此，让我们从示例图像开始。只需在灰度模式下加载图像并找到其完整的直方图即可。

"""
# %%
"""
绘制直方图：
    1.matplotlib绘图
    2.opencv绘图
"""
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('data/images/hot.png',0)
# cv中的直方图计算
hist = cv.calcHist([img],[0],None,[256],[0,256])
# numpy中的直方图计算
hist, bins = np.histogram(img.ravel(), 256, [0, 256])

plt.hist(img.ravel(), 256, [0,256]),plt.show()

#%% matplotlib的法线图

img = cv.imread('data/images/1_copy.png')
color = ('b', 'g', 'r')

for i, col in enumerate(color):
    histr = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0,256])

plt.show()

#%% opencv绘制直方图
img = cv.imread('data/images/1_copy.png')

mask = np.zeros(img.shape[:2], np.uint8)
mask[100:300, 400:700] = 255
masked_img = cv.bitwise_and(img,img,mask = mask)

hist_full = cv.calcHist([img],[0],None,[256],[0,256])
hist_mask = cv.calcHist([img],[0],mask,[256],[0,256])
plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask,'gray')
plt.subplot(223), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0,256])
plt.show()


#%% 直方图均衡
"""
用于改善图像的对比度（直方图进行拉伸到两端操作）
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('data/images/football.png')

hist,bins = np.histogram(img.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()
plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()


cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')
img2 = cdf[img]

plt.plot(cdf_normalized, color = 'b')
plt.hist(img2.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()

#%% 直方图均衡

img = cv.imread('data/images/d.png', 0)
equ = cv.equalizeHist(img)
res = np.hstack((img, equ))

cv.imshow('res', res)
cv.waitKey(0)
cv.destroyAllWindows()

#%% CLAHE(对比度受限的自适应直方图均衡)
# 图像被分割成小块，默认是8*8，对每个块进行均衡

img = cv.imread('data/images/d.png', 0)
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl1 = clahe.apply(img)

res2 = np.hstack((res, cl1))
cv.imshow('cl1', res2)
cv.waitKey(0)
cv.destroyAllWindows()