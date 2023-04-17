"""
入门操作
    图像入门
    视频入门
    绘图功能
    鼠标作为画笔
    作为调色板的跟踪栏
"""

#%%
"""
图像入门
  ███                                             
 ▒▒▒                                              
 ████  █████████████    ██████    ███████  ██████ 
▒▒███ ▒▒███▒▒███▒▒███  ▒▒▒▒▒███  ███▒▒███ ███▒▒███
 ▒███  ▒███ ▒███ ▒███   ███████ ▒███ ▒███▒███████ 
 ▒███  ▒███ ▒███ ▒███  ███▒▒███ ▒███ ▒███▒███▒▒▒  
 █████ █████▒███ █████▒▒████████▒▒███████▒▒██████ 
▒▒▒▒▒ ▒▒▒▒▒ ▒▒▒ ▒▒▒▒▒  ▒▒▒▒▒▒▒▒  ▒▒▒▒▒███ ▒▒▒▒▒▒  
                                 ███ ▒███         
                                ▒▒██████          
                                 ▒▒▒▒▒▒           

"""
import cv2 as cv

# openCV读取图像
img = cv.imread('data/images/1.jpg')

# 显示图像
# 能调整窗口大小
cv.namedWindow('img', cv.WINDOW_NORMAL)
cv.imshow('img', img)
# 键盘绑定函数，它的参数是以毫秒为单位的时间。
# 该函数为任意键盘事件等待指定毫秒。如果你在这段时间内按下任意键，程序将继续。
# 如果传的是 0，它会一直等待键盘按下。它也可以设置检测特定的击键，例如，按下键 a 等，
cv.waitKey(0)
# 简单的销毁我们创建的所有窗口
cv.destroyAllWindows()

#%%
# 保存图像
cv.imwrite('data/images/1_copy.png', img)

#%%
"""
下面的程序以灰度模式读取图像，显示图像，
如果你按下 's‘ 会保存和退出图像，或者按下 ESC 退出不保存。
"""

img = cv.imread('data/images/1.jpg', 0)
cv.imshow('img', img)

k = cv.waitKey(0) & 0xFF
if k == 27: # ESC退出
    cv.destroyAllWindows()
elif k == ord('s'):
    cv.imwrite('data/images/2.png', img)
    cv.destroyAllWindows()

#%%
# matplotlib显示图像
from matplotlib import pyplot as plt

cv.imread('data/images/1.jpg', 0)
plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([]) # 隐藏 X 和 Y 轴的刻度值
plt.show()


#%%
"""
视频入门
              ███      █████                  
             ▒▒▒      ▒▒███                   
 █████ █████ ████   ███████   ██████   ██████ 
▒▒███ ▒▒███ ▒▒███  ███▒▒███  ███▒▒███ ███▒▒███
 ▒███  ▒███  ▒███ ▒███ ▒███ ▒███████ ▒███ ▒███
 ▒▒███ ███   ▒███ ▒███ ▒███ ▒███▒▒▒  ▒███ ▒███
  ▒▒█████    █████▒▒████████▒▒██████ ▒▒██████ 
   ▒▒▒▒▒    ▒▒▒▒▒  ▒▒▒▒▒▒▒▒  ▒▒▒▒▒▒   ▒▒▒▒▒▒  

"""

#%%
import numpy as np
import cv2 as cv

# 从相机捕获视频 选择相机
cap = cv.VideoCapture(0)
# 播放本地视频
# cap = cv.VideoCapture('data/videos/1.avi')

while(True):
    # 逐帧捕获
    ret, frame = cap.read()
    # 对帧操作
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    # 显示帧
    cv.imshow('frame', gray)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# 按q退出后，释放资源
cap.release()
cv.destroyAllWindows()

#%% 捕获并保存视频

cap = cv.VideoCapture(0)
"""声明编码器
FourCC 是用于指定视频解码器的 4 字节代码。这里 fourcc.org 是可用编码的列表。它取决于平台，下面编码就很好。
In Fedora: DIVX, XVID, MJPG, X264, WMV1, WMV2. (XVID 是最合适的. MJPG 结果比较大. X264 结果比较小)
In Windows: DIVX (还需要测试和添加跟多内容)
"""
fourcc = cv.VideoWriter_fourcc(*'XVID')

"""
创建 VideoWrite 对象
cv::VideoWriter参数：
    const String & 	filename,
    int 	fourcc,
    double 	fps,
    Size 	frameSize,
    bool 	isColor = true
"""
out = cv.VideoWriter('data/videos/output.avi', fourcc, 20.0, (640, 480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        # 0 表示绕 x 轴翻转，正值（例如 1）表示绕 y 轴翻转。负值（例如 -1）表示围绕两个轴翻转。
        frame = cv.flip(frame, 0)
        # 写入翻转好的帧
        out.write(frame)
        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
out.release()
cv.destroyAllWindows()


#%%

"""
绘图入门
                      ███              █████   
                     ▒▒▒              ▒▒███    
 ████████   ██████   ████  ████████   ███████  
▒▒███▒▒███ ▒▒▒▒▒███ ▒▒███ ▒▒███▒▒███ ▒▒▒███▒   
 ▒███ ▒███  ███████  ▒███  ▒███ ▒███   ▒███    
 ▒███ ▒███ ███▒▒███  ▒███  ▒███ ▒███   ▒███ ███
 ▒███████ ▒▒████████ █████ ████ █████  ▒▒█████ 
 ▒███▒▒▒   ▒▒▒▒▒▒▒▒ ▒▒▒▒▒ ▒▒▒▒ ▒▒▒▒▒    ▒▒▒▒▒  
 ▒███                                          
 █████                                         
▒▒▒▒▒                                          

"""

