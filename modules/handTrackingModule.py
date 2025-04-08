import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'

import cv2
import mediapipe as mp
import time

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
        landmarkList = []
        if self.results and self.results.multi_hand_landmarks: # ensuring res exists
            if handNo < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c, = img.shape 
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarkList.append([id, cx, cy])
                    if draw:
                        # Example: Draw circles on all landmarks 
                        # cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                        # Example: Highlight a specific landmark (e.g., wrist)
                        if id == 0:
                            cv2.circle(img, (cx, cy), 15, (255,0,255), cv2.FILLED)
        
        return landmarkList

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
        landmarkList = detector.findPosition(img, draw=True)

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
