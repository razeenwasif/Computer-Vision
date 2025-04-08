# This is the main python file for the virtual mouse project.
import os 
import sys
import cv2 
import time 
import autopy
import numpy as np

# --- Add project root to Python path ---
# Get the directory containing this script (virtual-mouse)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (your_project_folder)
project_root = os.path.dirname(current_dir)
# Add the project root to the start of the Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modules.handTrackingModule import HandDetector 


camWidth, camHeight = 640, 480
frameReduction = 100 
screenWidth, screenHeight = autopy.screen.size()
smoothener = 5
prevLocX, prevLocY = 0, 0
currLocX, currLocY = 0, 0

capture = cv2.VideoCapture(0)
capture.set(3,camWidth), capture.set(4, camHeight)
prevTime = 0

detector = HandDetector(maxHands = 1)

while True:
    # Find hand landmarks
    success, img = capture.read()
    img = detector.findHands(img)
    landmarkList, bbox = detector.findPosition(img)
    
    # Get the tip of the index and middle fingers
    if len(landmarkList) != 0:
        x1, y1 = landmarkList[8][1:] # index
        x2, y2 = landmarkList[12][1:] # middle
        
        # print(x1,y1,x2,y2)

        # Check which fingers are up
        fingers = detector.fingersUp()
        #print(fingers)

        cv2.rectangle(img, (frameReduction, frameReduction), 
                          (camWidth - frameReduction, camHeight - frameReduction),
                          (255,0,255), 2)

        # Index Finger -> Moving mode 
        if fingers[1] == 1 and fingers[2] == 0:
            # Convert coordinates 
            x3 = np.interp(x1, (frameReduction,camWidth-frameReduction), 
                           (0,screenWidth))
            y3 = np.interp(y1, (frameReduction,camHeight-frameReduction), 
                           (0,screenHeight))

            # Smoothen the values
            currLocX = prevLocX + (x3 - prevLocX) / smoothener
            currLocY = prevLocY + (y3 - prevLocY) / smoothener

            # Moving the mouse
            autopy.mouse.move(screenWidth - currLocX, currLocY)
            cv2.circle(img, (x1,y1), 15, (255,0,255), cv2.FILLED)
            prevLocX, prevLocY = currLocX, currLocY

        # Both index and middle fingers are up -> clicking mode 
        if fingers[1] == 1 and fingers[2] == 1:
            # Find distance between fingers 
            length, img, lineInfo = detector.findDistance(8, 12, img)
            # distance threshold for clicking = 30
            if length < 30: 
                cv2.circle(img, (lineInfo[4],lineInfo[5]), 15, (0,255,0), cv2.FILLED)
                autopy.mouse.click()
    
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
        cv2.imshow("Virtual Mouse", img)
    else:
        print(f"Warning: Frame is None even though success was true")
        continue # skip processing this frame


    # --- wait for key press --- 
    key = cv2.waitKey(1) & 0xFF # 0xFF for cross platform compatibility
    if key == ord('q'):
        print("exiting")
        break 

