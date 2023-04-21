import cv2
import numpy as np

from pose import Pose

class HandleCapture:
    def __init__(self):
        # 创建肢体关键点
        self.pose = Pose()
        pass

    def handleVideoCapture(self):
        # 创建摄像头
        # cap = cv2.VideoCapture(r'E:\cao\pytorchProject\data\videos\kun.mp4')
        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output3.mp4', fourcc, 20, (1280, 720))
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print('cant read image')
                break
            # 处理肢体关键点
            # frame = cv2.flip(frame, flipCode=-1)
            # 在背景上绘制
            bg = np.full((720, 1280, 3), fill_value=(142, 133, 133), dtype=np.uint8)

            self.pose.process(frame, bg)
            cv2.imshow('img', frame)
            # cv2.imshow('img', bg)
            # out.write(bg)

            key = cv2.waitKey(25)
            if key == ord('q'):
                break
        out.release()

        cap.release()
        cv2.destroyAllWindows()

cap = HandleCapture()
cap.handleVideoCapture()
