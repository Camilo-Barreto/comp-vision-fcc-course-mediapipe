import cv2
import mediapipe as mp
import numpy as np
import time
import os
import sys
sys.path.append("HandTracking")
import HandTrackingModule as htm # type: ignore

####################
brushThickness = 7
eraserThickness = 100
####################

folderPath = "AI Virtual Painter Project/Header"
myList = os.listdir(folderPath)
overlayList = []

for imgPath in myList:
    image = cv2.imread(f'{folderPath}/{imgPath}')
    overlayList.append(image)

header = overlayList[0]
# Set color for the circle shown on the finger
drawColor = (0, 255, 0)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
pTime = 0

detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), dtype="uint8")

while True:
    # 1. Import the image and flip it on the y axis
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. Find hand landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # Tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingersUp()

        # 4. If selection mode - 2 fingers are up
        if fingers[1] and fingers[2]:
            print("Selection mode")
            # Set xp and yp for drawing back to 0
            xp, yp = 0, 0

            ######################################################
            # Draw selection partition lines for debugging       #
            ######################################################
            # cv2.line(img, (250, 0), (250, 720), (0, 0, 255), 3)
            # cv2.line(img, (450, 0), (450, 720), (0, 0, 255), 3)
            #
            # cv2.line(img, (500, 0), (500, 720), (0, 0, 255), 3)
            # cv2.line(img, (750, 0), (750, 720), (0, 0, 255), 3)
            #
            # cv2.line(img, (800, 0), (800, 720), (0, 0, 255), 3)
            # cv2.line(img, (1000, 0), (1000, 720), (0, 0, 255), 3)
            #
            # cv2.line(img, (1050, 0), (1050, 720), (0, 0, 255), 3)
            # cv2.line(img, (1250, 0), (1250, 720), (0, 0, 255), 3)
            ######################################################

            # Check if finger is in the header
            if y1 < 150:
                if 250 < x1 <450:
                    header = overlayList[0]
                    drawColor = (0, 255, 0)
                elif 500 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 800 < x1 < 1000:
                    header = overlayList[2]
                    drawColor = (0, 0, 255)
                elif 1050 < x1 < 1250:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
                    
            cv2.rectangle(img, (x1, y1-25), (x2, y2+25), drawColor, cv2.FILLED)


        # 5. If drawing mode - Index finger is up
        if fingers[1] and fingers[2]==False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing mode")
            # If just started to draw take the current pos as the previous
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            
            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            # Set current pos as the previous for next iteration
            xp, yp = x1, y1

    # To draw on the video
    # Convert img to grayscle
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    # Apply binary threshold
    _, imgInv = cv2.threshold(imgGray, 10, 255, cv2.THRESH_BINARY_INV)
    # Convert inverse to BGR
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    # AND to merge drawing from inverse to img
    img = cv2.bitwise_and(img, imgInv)
    # OR to add the color from imgCanvas to img
    img = cv2.bitwise_or(img, imgCanvas)
    # cv2.imshow("Inv", imgInv)

    # Overlay the header
    img[0:150, 0:1280] = header

    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)

    # Get the fps and display it
    cTime = time.time()
    fps = 1/ (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (10, 180), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Image", img)
    #cv2.imshow("Canvas", imgCanvas)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
