import cv2
import mediapipe as mp
import utils

class Pose:
    def __init__(self):
        mpPose = mp.solutions.pose
        # 加载模型
        self.pose = mpPose.Pose()

        # 关键点
        self.landmark = []

        # 识别图像
        self.image = None
        pass

    def process(self, img, bg):
        self.img = img
        pose = self.pose
        # BGR转RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = pose.process(imgRGB)
        # print(res.pose_landmarks)

        # 绘制关键点连线draw_landmarks
        """
            参数一，原画
            参数二，关键点
            参数三，指定连接点
            参数四，连接点样式
            参数五，连接线样式
        """
        pointStyle = mp.solutions.drawing_utils.DrawingSpec(color=(0,0,255), thickness=2)
        lineStyle = mp.solutions.drawing_utils.DrawingSpec(color=(255,0,0), thickness=2)

        conn = mp.solutions.pose_connections
        mp.solutions.drawing_utils.draw_landmarks(
            img,
            res.pose_landmarks,
            conn.POSE_CONNECTIONS,
            pointStyle,
            lineStyle
        )

        # 绘制关键点的序号
        landmark = res.pose_landmarks.landmark
        if not landmark:
            return
        self.landmark = landmark
        h, w, _ = imgRGB.shape
        for idx, lm in enumerate(landmark):
            x = int(lm.x*w)
            y = int(lm.y*h)
            cv2.putText(img,
                        str(idx),
                        org=(x-15, y-5),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.3,
                        color = (0,255,0),
                        thickness=1)

        self.armBuildCnt()
        pass

    # 计算手臂弯曲角度
    def armBuildCnt(self):


        point12 = self.point(12)
        point14 = self.point(14)
        point16 = self.point(16)
        angle = utils.calculate_angle(point12, point14, point16)
        print(angle)
        pass


    def point(self, index):
        h, w = self.img.shape[:2]
        lm = self.landmark[index]
        x = lm.x * h
        y = lm.y * h
        return x, y