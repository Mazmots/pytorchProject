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