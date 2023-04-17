"""
  ████████
 ███▒▒▒▒███
▒███   ▒███
▒▒█████████
 ▒▒▒▒▒▒▒███
 ███   ▒███
▒▒████████
 ▒▒▒▒▒▒▒▒
轮廓
"""
import copy

#%%
import numpy as np
import cv2 as cv

img = cv.imread('data/images/canny.png')
imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)

"""
第三个参数ContourApproximationModes
CV_CHAIN_APPROX_NONE：将所有的连码点，转换成点。
CV_CHAIN_APPROX_SIMPLE：压缩水平的、垂直的和斜的部分，也就是，函数只保留他们的终点部分。
CV_CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS：使用the flavors of Teh-Chin chain近似算法的一种。
"""
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


"""
void cv::drawContours	(	
    InputOutputArray 	image,    # 源图像
    InputArrayOfArrays 	contours, # 应该作为Python列表传递的轮廓
    int 	contourIdx,           # 轮廓的索引  在绘制单个轮廓时很有用。要绘制所有轮廓，请传递-1
    const Scalar & 	color,        # 颜色
    int 	thickness = 1,        # 厚度
    int 	lineType = LINE_8,
    InputArray 	hierarchy = noArray(),
    int 	maxLevel = INT_MAX,
    Point 	offset = Point() 
    )		

"""
draw_contours = cv.drawContours(img, contours, -1, (0, 255, 0), 2)
cv.imshow('', draw_contours)
cv.waitKey(0)
cv.destroyAllWindows()



#%%
"""
轮廓特征
学习找到轮廓的不同特征，如区域，周长，边界矩形等。
"""

#%% 矩

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import copy

img = cv.imread('data/images/f.png')

imggray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imggray, 127, 255, cv.THRESH_OTSU)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cnt = contours[0]

# 直角矩形
# (x, y)为矩形左上角的坐标， (w, h)是矩形的宽和高
x, y, w, h = cv.boundingRect(cnt)
# 直边外接矩形
img_boundingRect = cv.rectangle(copy.deepcopy(img), (x, y), (x+w, y+h), (0, 255, 0), 3)

# 旋转矩形
rect = cv.minAreaRect(cnt)
box = cv.boxPoints(rect)
box = np.int0(box)
cv_draw_contours = cv.drawContours(copy.deepcopy(img), [box], 0, (255, 0, 0), 2)

# 最小外圆
(x,y),radius = cv.minEnclosingCircle(cnt)
center = (int(x), int(y))
radius = int(radius)
circle = cv.circle(copy.deepcopy(img), center, radius, (0, 255, 0), 2)

# 拟合椭圆
ellipse = cv.fitEllipse(cnt)
cv_ellipse = cv.ellipse(copy.deepcopy(img), ellipse, (0, 255, 0), 2)

# 拟合线
rows,cols = img.shape[:2]
[vx,vy,x,y] = cv.fitLine(cnt, cv.DIST_L2,0,0.01,0.01)
lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx)+y)
cv_line = cv.line(copy.deepcopy(img), (cols - 1, righty), (0, lefty), (0, 255, 0), 2)

plt.subplot(231),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])

plt.subplot(232),plt.imshow(img_boundingRect),plt.title('boundingRect')
plt.xticks([]), plt.yticks([])

plt.subplot(233),plt.imshow(cv_draw_contours),plt.title('旋转矩形')
plt.xticks([]), plt.yticks([])

plt.subplot(234),plt.imshow(circle),plt.title('circle')
plt.xticks([]), plt.yticks([])

plt.subplot(235),plt.imshow(cv_ellipse),plt.title('cv_ellipse')
plt.xticks([]), plt.yticks([])

plt.subplot(236),plt.imshow(cv_line),plt.title('cv_line')
plt.xticks([]), plt.yticks([])

plt.show()


"""
凸包检测

"""
import cv2 as cv

# 读取图像
src = cv.imread('data/images/f.png')
# 转换为灰度图像
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 二值化
ret, thre = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
# 获取结构元素
k = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
# 开操作
cv.morphologyEx(thre, cv.MORPH_OPEN, k)
# 轮廓发现
counters, hierarchy = cv.findContours(thre, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# 在原图上绘制轮廓，以方便和凸包对比，发现凸缺陷
cv.drawContours(src, contours, -1, (0,255,0), 3)
