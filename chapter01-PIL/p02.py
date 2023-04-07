import random
from PIL import Image, ImageDraw, ImageFont
"""
PIL生成随机码
实现步骤:
1、生成随机数函数
其中第48~57号为0~9十个阿拉伯数字;
65~90号为26个大写英文字母;
97~122号为26个小写英文字母
2、生成随机的字体颜色
3、生成随机的背景颜色
4、画图生成验证码
"""
# 1、生成随机数函数
def get_coder():
    codes = [code for code in range(48, 57)]
    codes += [code for code in range(65, 91)]
    codes += [code for code in range(95, 123)]
    code = codes[random.randint(0, len(codes))]
    print(code)
    # 0到255以内的整数
    return chr(code)
# 2、生成随机的字体颜色
# 生成的 矩阵随机即可。
# 取值 120 - 230即可。和字体分开
def get_font_color():
    return (random.randint(120, 230), random.randint(120, 230),
random.randint(120, 230))
# 3、生成随机的背景颜色
def get_back_color():
    return (random.randint(0, 120), random.randint(0, 120), random.randint(0,120))
# 4、画图生成验证码
# 4.1 创建Image对象
w = 240
h = 120
img = Image.new("RGB", size=(w, h),color=(255, 255, 255))
# 4.2 创建画笔
draw = ImageDraw.Draw(img)
# 4.3 创建字体
font = ImageFont.truetype(r"C:\Windows\Fonts\comic.ttf",
size=66)
# 背景颜色随机
for y in range(h):
    for x in range(w):
        draw.point((x, y), fill=get_back_color())
# 绘制随机数
for i in range(4):
    draw.text((10 + w // 4 * i, 18), text=get_coder(), fill=get_font_color(),
font=font)
img.show()
