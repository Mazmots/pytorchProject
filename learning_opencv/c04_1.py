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
#%%
import cv2 as cv

# 读取图像
src1 = cv.imread('data/images/star.png')
# 转换为灰度图像
gray = cv.cvtColor(src1, cv.COLOR_BGR2GRAY)
# 二值化
ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
# 获取结构元素
k = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
# 开操作
binary = cv.morphologyEx(binary, cv.MORPH_OPEN, k)
# 轮廓发现
contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# 在原图上绘制轮廓，以方便和凸包对比，发现凸缺陷
cv.drawContours(src1, contours, -1, (0, 225, 0), 3)
for c in range(len(contours)):
    # 是否为凸包
    ret = cv.isContourConvex(contours[c])
    # 凸缺陷
    # 凸包检测，returnPoints为false的是返回与凸包点对应的轮廓上的点对应的index
    hull = cv.convexHull(contours[c], returnPoints=False)
    defects = cv.convexityDefects(contours[c], hull)
    print('defects', defects)

    for j in range(defects.shape[0]):
        s, e, f, d = defects[j, 0]
        start = tuple(contours[c][s][0])
        end = tuple(contours[c][e][0])
        far = tuple(contours[c][f][0])
        # 用红色连接凸缺陷的起始点和终止点
        cv.line(src1, start, end, (0, 0, 225), 2)
        # 用蓝色最远点画一个圆圈
        cv.circle(src1, far, 5, (225, 0, 0), -1)

# 显示
cv.imshow("result", src1)
cv.waitKey(0)
cv.destroyAllWindows()


#%% 使用 cv.matchShapes() 比较数字或字母的图像

import numpy as np
import cv2# 读取两张待比较的图片
img1 = cv2.imread('data/images/star.png')
img2 = cv2.imread('data/images/star2.png')# 输入两张图片的轮廓
contours1, _ = cv2.findContours(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours2, _ = cv2.findContours(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# 计算两个轮廓之间的距离
dist = cv2.matchShapes(contours1[0], contours2[0], cv2.CONTOURS_MATCH_I1, 0)
# 打印距离
print("距离：", dist)



#%%
import cv2
import imutils
import numpy as np

def c_and_b(arg):
    ''''''
    cnum = cv2.getTrackbarPos(trackbar_name1, wname)
    bnum = cv2.getTrackbarPos(trackbar_name2, wname)
    #print(bnum)
    cimg = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            lst = 0.1*cnum*img[i, j] + bnum
            cimg[i, j] = [int(ele) if ele < 255 else 255 for ele in lst]
    cv2.imshow(wname, imutils.resize(cimg, 800))

wname = 'brightness and contrast'
trackbar_name1 = 'contrast'
trackbar_name2 = 'brightness'
img = cv2.imread(r"E:\cao\pytorchProject\data\images\football.png")
height, width = img.shape[:2]
img = cv2.resize(img, (int(width/height*400), 400), interpolation=cv2.INTER_CUBIC)

cv2.namedWindow(wname)
cv2.createTrackbar(trackbar_name1, wname, 10, 20, c_and_b)
cv2.createTrackbar(trackbar_name2, wname, 0, 100, c_and_b)

c_and_b(0)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()