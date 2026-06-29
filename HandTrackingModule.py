import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

class HandDetector:
    def __init__(self, maxHands=2):
        self.maxHands = maxHands
        base_options = python.BaseOptions(model_asset_path=r'C:\Users\Pc Booster\PycharmProjects\PythonProject1\HandTracking\hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=self.maxHands,
            running_mode=vision.RunningMode.VIDEO
        )
        self.hands = vision.HandLandmarker.create_from_options(options)

    def findHands(self, img, cap=None):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        timestamp = int(time.time() * 1000)
        results = self.hands.detect_for_video(mp_image, timestamp)

        lmList = []
        if results.hand_landmarks:
            for handlms in results.hand_landmarks:
                h, w, _ = img.shape
                for a, b in CONNECTIONS:
                    x1, y1 = int(handlms[a].x * w), int(handlms[a].y * h)
                    x2, y2 = int(handlms[b].x * w), int(handlms[b].y * h)
                    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                for id, lm in enumerate(handlms):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 4, (255, 0, 255), cv2.FILLED)
                    lmList.append([id, cx, cy])


        return img, lmList


def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        img, lmList = detector.findHands(img, cap)
        if len(lmList) != 0:
            print(lmList)



        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()