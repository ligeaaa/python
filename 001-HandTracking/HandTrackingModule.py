import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands = 2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        # 创建一个手的实体类来存储手部信息
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, 1, self.detectionCon, self.trackCon)
        # 创建一个用于绘制手部关键点和连接线的实体类
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw = True):
        # 将捕获的图像从BGR颜色空间转换为RGB颜色空间
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 处理图像以检测手部信息
        self.results = self.hands.process(imgRGB)

        # 检查是否检测到了手部关键点
        if self.results.multi_hand_landmarks:
            # 遍历每个检测到的手
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # 在图像上绘制手部关键点和连接线
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):

        lmList = []

        # 检查是否检测到了手部关键点
        if self.results.multi_hand_landmarks:
            # 存储对应的手，此处默认handNo=0
            myHand = self.results.multi_hand_landmarks[handNo]
            # 获得手部21点的id和其对应的坐标
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # 将参数添加入lmList
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        return lmList

def main():
    pTime = 0
    cTime = 0
    # 打开默认摄像头
    cap = cv2.VideoCapture(0)
    # 生成实体类
    detector = handDetector()

    while True:
        # 从摄像头捕获一帧图像
        success, img = cap.read()
        # 调用实体类的相关函数
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        # 在窗口中显示图像
        cv2.imshow("Image", img)

        # 按键等待1毫秒，允许窗口保持打开状态
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
