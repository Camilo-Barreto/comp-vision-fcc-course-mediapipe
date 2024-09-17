import cv2
import time
import os
import sys
sys.path.append("HandTracking")
import HandTrackingModule as htm # type: ignore

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Retrieve the images folder
folderPath = "Finger Counter Project/Finger images"
myList = os.listdir(folderPath)
print(myList)

# Retrieve all individual images
overlayList = []
for imgPath in myList:
    image = cv2.imread(f'{folderPath}/{imgPath}')
    overlayList.append(image)

pTime = 0
detector = htm.handDetector(detectionCon=0.75)
# Mediapipe landmarks for finger tips
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()

    # Get the data for the hands
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)

    # Only perform the following when a hand is detected
    if len(lmList) != 0:
        fingers = []
        # If tip of finger y cord is less than middle joint y cord, the finger is open
        # The img pixels y starts from the top at 0 and increments to the bottom
        # Detection points is 2 less than the tip. Eg. for 8 (index tip) detection point is 6
        
        # Special case for thumb, check direction of finger whether left or right of -1 point
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            # Detection points is 2 less than the tip. Eg. for 8 (index tip) detection point is 6
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)

        # Convert and display an image of 200 x 200 for the fingers
        # For 0 fingers, 0-1= -1 (gets the last image)
        # For 1 finger,  1-1= 0 (gets the first image) and so on
        h, w, c = overlayList[totalFingers-1].shape
        img[0:h, 0:w] = overlayList[totalFingers-1]

        cv2.rectangle(img, (20, 255), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 
                    10, (255, 0, 0), 25)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (240, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break