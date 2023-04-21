import cv2 as cv
import numpy as np

img = np.zeros((512, 512, 3), np.uint8)


def callback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print(f'{event},{x},{y}')
        cv.circle(img,(x,y),50,(255,0,0),2)
    pass

def mouseDrawing():

    cv.namedWindow('board')

    cv.setMouseCallback('board', callback)

    # cv.imshow('img', img)
    cv.waitKey(0)
    cv.destroyAllWindows()



mouseDrawing()
