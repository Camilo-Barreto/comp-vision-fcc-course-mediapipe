import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

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
        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            # Get landmarks data
            for id, lm in enumerate(myHand.landmark):
                # Height, width, channels 
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)

        return lmList

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