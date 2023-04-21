#%% 直接读取图片
import cv2 as cv
import numpy as np

img = cv.imread(r'data/images/football.png')

h, w, c = img.shape
print(h,w,c)

b, g, r = cv.split(img)

merge = cv.merge((r,  b))
cv.imshow('r', r)
cv.imshow('g', g)
cv.imshow('b', b)
cv.imshow('merge', merge)
cv.waitKey(0)
cv.destroyAllWindows()


#%% 矩阵创建图片
import cv2 as cv
import numpy as np
data = np.full((300, 600, 3), fill_value=(255, 111, 222), dtype=np.uint8)

cv.imshow('data', data)
cv.waitKey(0)
cv.destroyAllWindows()


#%% 捕获视频

cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    cv.imshow('f', frame)
    if cv.waitKey(1) & 0xff ==ord('q'):
        break

cap.release()
cv.destroyAllWindows()

#%% 图像加边框
import cv2 as cv
import numpy as np
img = cv.imread(r'/data/images/canny.png')

padding = 20
h,w,c = img.shape
#
# border = cv.copyMakeBorder(img, padding, padding, padding, padding, cv.BORDER_REFLECT, value=(255, 0, 0))

r = (255,0,0)
b = (0,0,255)

# img[:padding, :] = r
# img[h-padding:h, :] = r
#
# img[:, :padding] = b
# img[:, w-padding:w] = b
#
# # cv.imshow('img', border)
# cv.imshow('img', img)
# cv.waitKey(0)
# cv.destroyAllWindows()


new_img = np.full((h + 2*padding, w + 2*padding, c), fill_value=(r), dtype=np.uint8)
new_img[padding:padding + h, padding:padding + w] = img

cv.imshow('new_img', new_img)
cv.waitKey(0)
cv.destroyAllWindows()


#%% 图像加logo

img = cv.imread(r'data/images/football.png')
logo = cv.imread(r'data/images/hot.png')
h,w,_ = logo.shape

logo_s = cv.resize(logo,dsize=(50,int(50*h/w)))
h,w,_ = logo_s.shape

img[:h, :w] = logo_s

cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()


#%% 绘制几何图形

img = cv.imread('data/images/canny.png')

color_r = (255,0,0)
# cv.line(img, (100,100), (300,300), color = color_r, thickness=2)
# cv.rectangle(img, (100,100), (300,300), color = color_r, thickness=2)
# cv.circle(img, (200,200), 100, color=color_r, thickness=-1)

pts = np.array([(100,200), (200,400), (400, 200), (50,50)])
cv.polylines(img, [pts], isClosed=True,color = color_r,thickness=3)

cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()