import numpy as np
from PIL import Image, ImageDraw

img = Image.open('img01.png')
print(img.size)

imgData = np.asarray(img)

print(imgData)
print(img.mode)
print(imgData.shape)

# 加水印
draw = ImageDraw.Draw(img)
# 字体颜色
fillColor = (255, 0, 0)

text = 'print text on PIL Image'
position = (200, 100)

draw.text(position, text, fill=fillColor)
img.save('test.png')




#%%
"""
函数说明：	im.convert(mode, parms**)
参数说明：
			（1）mode：指的是要转换成的图像模式；
			（2）parms：其他可选参数。如：matrix、dither 等。
			其中，最关键的参数是 mode，其余参数无须关心。
"""
from PIL import Image
import matplotlib.pyplot as plt

im = Image.open("img01.png")
print(im.mode)		# RGB
im1 = im.convert('1')
im2 = im.convert('L')
im3 = im.convert('P')
im4 = im.convert('RGB')
im5 = im.convert('RGBA')
im6 = im.convert('CMYK')
im7 = im.convert('YCbCr')
# im8 = im.convert('LAB')			# ValueError: conversion from RGB to LAB not supported
im9 = im.convert('HSV')
im10 = im.convert('I')
im10.show()
im11 = im.convert('F')
im11.show()
######################################################################
# 绘图
im_list = [im, im1, im2, im3, im4, im5, im6, im7, im9, im10, im11]
for i, j in enumerate(im_list):
    plt.subplot(3, 4, i+1)
    plt.title(['raw', '1', 'L', 'P', 'RGB', 'RGBA', 'CMYK', 'YCbCr', 'HSV', 'I', 'F'][i])
    plt.axis('off')
    plt.imshow(j)
plt.show()

"""
plt.subplot(3,4,1), plt.imshow(im), plt.title('raw'), plt.axis('off')
plt.subplot(3,4,2), plt.imshow(im1), plt.title('1'), plt.axis('off')
plt.subplot(3,4,3), plt.imshow(im2), plt.title('L'), plt.axis('off')
plt.subplot(3,4,4), plt.imshow(im3), plt.title('P'), plt.axis('off')
plt.subplot(3,4,5), plt.imshow(im4), plt.title('RGB'), plt.axis('off')
plt.subplot(3,4,6), plt.imshow(im5), plt.title('RGBA'), plt.axis('off')
plt.subplot(3,4,7), plt.imshow(im6), plt.title('CMYK'), plt.axis('off')
plt.subplot(3,4,8), plt.imshow(im7), plt.title('YCbCr'), plt.axis('off')
# plt.subplot(3,4,9), plt.imshow(im8), plt.title('LAB'), plt.axis('off')
plt.subplot(3,4,10), plt.imshow(im9), plt.title('HSV'), plt.axis('off')
plt.subplot(3,4,11), plt.imshow(im10), plt.title('I'), plt.axis('off')
plt.subplot(3,4,12), plt.imshow(im11), plt.title('F'), plt.axis('off')
plt.show()
"""

