import cv2
import mediapipe as mp
import time

# Create a video object
camera_index = 0
cap = cv2.VideoCapture(camera_index)

# Initialisations
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Varibles to detect frame rate. Previous Time and Current Time
pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the hand and create a results object. (pass in the rgb video frames)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    
    # Extract multiple hands from results
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Get landmarks data
            for id, lm in enumerate(handLms.landmark):
                # Height, width, channels 
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)

                if id == 4:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            # Draw landmarks on the hand (handLms) on the original BGR image
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Detect frame rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break