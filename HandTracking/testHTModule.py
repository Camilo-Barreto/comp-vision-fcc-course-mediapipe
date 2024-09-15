import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

# Varibles to detect frame rate. Previous Time and Current Time
pTime = 0
cTime = 0
# Create a video object
camera_index = 0
cap = cv2.VideoCapture(camera_index)

detector = htm.handDetector()
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        print(lmList[4])

    # Detect frame rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break