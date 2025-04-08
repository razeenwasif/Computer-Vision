import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'

import cv2
import mediapipe as mp
import time
import math

class HandDetector():
    def __init__(
            self, 
            mode=False, 
            maxHands=2, 
            modelComplexity=1, 
            detectionCon=0.5, 
            trackCon=0.5
    ):
        # Store parameters
        self.mode = mode 
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Initialize mediapipe hands 
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode, self.maxHands, self.modelComplexity,
            self.detectionCon, self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

        self.results = None # store results 

    def findHands(self, img, draw=True):
        """Finds hands in an image and draws landmarks if draw=True"""
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            if draw:
                for handlandmarks in self.results.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(img, handlandmarks,
                                               self.mpHands.HAND_CONNECTIONS)
        return img 
    
    def findPosition(self, img, handNo=0, draw=True):
        """Finds landmark positions for a specific hand"""
        xList, yList, bbox = [], [], []
        self.landmarkList = []
        if self.results and self.results.multi_hand_landmarks: # ensuring res exists
            if handNo < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c, = img.shape 
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    xList.append(cx)
                    yList.append(cy)
                    self.landmarkList.append([id, cx, cy])
                    if draw:
                        # Example: Draw circles on all landmarks 
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                        # Example: Highlight a specific landmark (e.g., wrist)
                        #if id == 0:
                        #    cv2.circle(img, (cx, cy), 15, (255,0,255), cv2.FILLED)
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bbox = xmin, ymin, xmax, ymax 

                if draw:
                    cv2.rectangle(img, (xmin-20, ymin-20), (xmax+20, ymax+20),
                                  (0,255,0), 2)
        
        return self.landmarkList, bbox

    def fingersUp(self):
        fingers = []
        
        # --- checks --- 
        if not self.landmarkList: # check if list empty
            print("Warning: landmarkList is empty in fingersUp")
            return fingers 

        # check if necessary landmarks exist 
        max_needed_index = max(self.tipIds[4], self.tipIds[4] - 2)
        if len(self.landmarkList) <= max_needed_index:
            print(f"Warning: landmark list too short ({len(self.landmarkList)})in fingersUp")
            return fingers 

        # Thumb (right handed) (left would be <)
        if self.landmarkList[self.tipIds[0]][1] > self.landmarkList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers 
        for id in range(1,5):
            if self.landmarkList[self.tipIds[id]][2] < self.landmarkList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        # --- checks --- 
        if not self.landmarkList or p1 >= len(self.landmarkList) or \
            p2 >= len(self.landmarkList):
                print(f"Warning: Invalid landmarks ({p1}, {p2}) or list for findDistance.")
                return 0, img, [0,0,0,0,0,0]

        x1, y1 = self.landmarkList[p1][1:]
        x2, y2 = self.landmarkList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2 
        
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255,0,255), t)
            cv2.circle(img, (x1, y1), r, (255,0,255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255,0,255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0,0,255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1,y1,x2,y2,cx,cy]


# --- main loop using handDetector class ---
def main ():
    # --- create video object --- 
    capture_index = 0
    capture = cv2.VideoCapture(capture_index)

    detector = HandDetector(detectionCon=0.7)
    prevTime = 0

    # --- Check if camera opened successfully ---
    if not capture.isOpened():
        print(f"Error: Could not open video device at index {capture_index}.")
        # Try next index if failed
        capture_index = 1
        print(f"Trying index {capture_index}...")
        capture = cv2.VideoCapture(capture_index)
        if not capture.isOpened():
            print(f"Error: Could not open video device at index {capture_index} either.")
            exit()

    print(f"successfully opened camera at index {capture_index}")

    # --- run webcam (close with q) ---
    while True:
        # --- Read frame ---
        success, img = capture.read()

        if not success:
            print("Error: Failed to grab frame.")
            break
     
        # use the detector 
        img = detector.findHands(img, draw=True)
        landmarkList, bbox = detector.findPosition(img, draw=True)

        if len(landmarkList) != 0:
            # example: print position of the wrist (landmark 0)
            print(f"Wrist position: {landmarkList[0]}")
            # example: draw circle based on findPosition data
            wrist_id, wrist_x, wrist_y = landmarkList[0]
            cv2.circle(img, (wrist_x, wrist_y), 15, (255, 0, 255), cv2.FILLED)

        # --- fps calculation --- 
        currTime = time.time()
        if prevTime > 0: # ensures prevTime has been set in a prev iteration
            fps = 1/(currTime - prevTime)
            cv2.putText(
                img, f"FPS: {int(fps)}", (10,70), 
                cv2.FONT_HERSHEY_PLAIN, 3, 
                (255,0,255), 3
            )
        prevTime = currTime

        # --- Display the frame --- 
        if img is not None: # Additional check 
            cv2.imshow("Hand Tracking", img)
        else:
            print(f"Warning: Frame is None even though success was true")
            continue # skip processing this frame


        # --- wait for key press --- 
        key = cv2.waitKey(1) & 0xFF # 0xFF for cross platform compatibility
        if key == ord('q'):
            print("exiting")
            break 

    # --- Release resources --- 
    print("Releasing capture device...")
    capture.release()
    print("Destroying all windows...")
    cv2.destroyAllWindows()
    print("Done.")

# ///////////////////////////// ///////////////////////////////////

if __name__ == "__main__":
    main()
