import cv2
import mediapipe as mp
import time
import math

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.tipIds = [4, 8, 12, 16, 20]

        # Initialisations
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the hand and create a results object. (pass in the rgb video frames)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        
        # Extract multiple hands from results
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # Draw landmarks on the hand (handLms) on the original BGR image
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img
    
    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            # Get landmarks data
            for id, lm in enumerate(myHand.landmark):
                # Height, width, channels 
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)

        return self.lmList
    
    def fingersUp(self):
        fingers = []
        # If tip of finger y cord is less than middle joint y cord, the finger is open
        # The img pixels y starts from the top at 0 and increments to the bottom
        # Detection points is 2 less than the tip. Eg. for 8 (index tip) detection point is 6
        
        # Special case for thumb, check direction of finger whether left or right of -1 point
        # Use < when thumb is on the left side of the video and > for the opposite

        # USE > for all projects. USE < for Virtual Painter as the image is flipped
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            # Detection points is 2 less than the tip. Eg. for 8 (index tip) detection point is 6
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers


    def findDistance(self, img, p1, p2, draw=True):
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2)//2, (y1 + y2)//2

        if draw:
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)

        length = math.hypot(x2-x1, y2-y1)
        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    # Varibles to detect frame rate. Previous Time and Current Time
    pTime = 0
    cTime = 0
    # Create a video object
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)

    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
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

if __name__ == "__main__":
    main()