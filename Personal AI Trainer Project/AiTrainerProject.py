import cv2 
import numpy as np
import time
import sys
sys.path.append("Pose Estimation")
import PoseModule as pm # type: ignore

cap = cv2.VideoCapture("Personal AI Trainer Project/Videos/1.mp4")
# Use device main camera
# cap = cv2.VideoCapture(0)
detector = pm.poseDetector()
count = 0
dir = 0
pTime = 0

while True:
    success, img = cap.read()
    img = cv2.resize(img, (1280, 720))
    # img = cv2.imread("Personal AI Trainer Project/Videos/test.jpg")
    img = detector.findPose(img, draw=False)
    lmList = detector.findPosition(img, draw=False)

    # Only perform the following if pose is detected
    if len(lmList) != 0:
        # Points of interest
        # Right Arm 12 14, 16
        angle = detector.findAngle(img, 12, 14, 16)
        # Left  Arm 11 13 15
        # detector.findAngle(img, 11, 13, 15)

        # Convert the range in 2nd position between 0 to 100
        per = np.interp(angle, (210, 310), (0, 100))
        bar = np.interp(angle, (220, 310), (650, 100))

        # Check for the dumbbell curls. Add 0.5 for down to up motion
        # Set custom color for progress bar
        color = (255, 0, 255)
        if per == 100:
            color = (0, 255, 0) 
            if dir == 0:
                count += 0.5
                dir = 1

        # Add another 0.5 for up to down motion
        if per == 0:
            color = (0, 0, 255)
            if dir == 1:
                count +=0.5
                dir = 0

        # Display progress bar
        cv2.rectangle(img, (1100, 100), (1175, 650), color, 2)
        cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 2, color, 4)     

        # Display count of bicep curls inside a rectangle
        cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(count)}', (45, 670), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)     

    cTime = time.time()
    fps = 1/ (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
 