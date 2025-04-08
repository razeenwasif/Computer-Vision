import os 
import sys
import cv2 
import time 
import autopy
import numpy as np

# --- Add project root to Python path --- 
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- Path setup done ---

# --- Module Import ---
from modules.handTrackingModule import HandDetector
from modules.oneEuroFilter import OneEuroFilter 

# --- Constants ---
CAM_WIDTH, CAM_HEIGHT = 640, 480
FRAME_REDUCTION = 100 # for mapping area
CLICK_DISTANCE_THRESHOLD = 30 # Made constant, adjust as needed
INDEX_TIP_ID = 8
MIDDLE_TIP_ID = 12

# --- Screen and Camera Setup ---
screen_width, screen_height = autopy.screen.size()
prev_loc_x, prev_loc_y = 0, 0
curr_loc_x, curr_loc_y = 0, 0

capture = cv2.VideoCapture(0)
if not capture.isOpened(): 
    print("Error: Could not open camera.")
    exit()

capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

# --- Hand Detector ---
detector = HandDetector(maxHands=1) 

# --- One Euro Filter setup --- 
initial_time = time.time()
# filter params (tune later)
filter_min_cutoff = 0.1
filter_beta = 0.005
filter_d_cutoff = 1.0

one_euro_filter_x = OneEuroFilter(
    initial_time, screen_width / 2, min_cutoff=filter_min_cutoff,
    beta = filter_beta, d_cutoff = filter_d_cutoff
)
one_euro_filter_y = OneEuroFilter(
    initial_time, screen_height / 2, min_cutoff=filter_min_cutoff,
    beta = filter_beta, d_cutoff = filter_d_cutoff
)

# --- Timing ---
prev_time = 0

# --- Main Loop ---
while True:
    # 1. Read frame
    success, img = capture.read()
    if not success:
        print("Error: Failed to grab frame.")
        break
    
    currTime = time.time()

    # --- Optional: Flip image for mirror view ---
    img = cv2.flip(img, 1)

    # 2. Find hand landmarks
    img = detector.findHands(img, draw=False) # drawing 
    landmarkList, bbox = detector.findPosition(img, draw=False) # drawing true

    # Draw the active area boundary
    cv2.rectangle(img, (FRAME_REDUCTION, FRAME_REDUCTION), 
                      (CAM_WIDTH - FRAME_REDUCTION, CAM_HEIGHT - FRAME_REDUCTION),
                      (255, 0, 255), 2)

    # 3. Get the tip of the index and middle fingers if hand is detected
    if landmarkList: 
        # Ensure we have enough landmarks before accessing tips
        if len(landmarkList) > max(INDEX_TIP_ID, MIDDLE_TIP_ID):
            x1, y1 = landmarkList[INDEX_TIP_ID][1:]   # Index finger tip coordinates
            x2, y2 = landmarkList[MIDDLE_TIP_ID][1:] # Middle finger tip coordinates

            # 4. Check which fingers are up
            fingers = detector.fingersUp()
            # print(fingers) # Debug print

            # 5. Index Finger Up: Moving Mode
            if fingers[1] == 1 and fingers[2] == 0:
                # Convert coordinates to screen space within the reduced frame
                x_mapped = np.interp(x1, (FRAME_REDUCTION, CAM_WIDTH - FRAME_REDUCTION), (0, screen_width))
                y_mapped = np.interp(y1, (FRAME_REDUCTION, CAM_HEIGHT - FRAME_REDUCTION), (0, screen_height))

                # Smoothen the values
                smooth_x = one_euro_filter_x(currTime, x_mapped)
                smooth_y = one_euro_filter_y(currTime, y_mapped)

                # Move Mouse
                autopy.mouse.move(smooth_x, smooth_y) 
                # Feedback circle
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)

            # 6. Both Index and Middle Fingers Up: Clicking Mode
            if fingers[1] == 1 and fingers[2] == 1:
                # Find distance between index and middle finger tips
                length, img, lineInfo = detector.findDistance(INDEX_TIP_ID, MIDDLE_TIP_ID, img)
                
                # Click mouse if distance is short enough
                if length < CLICK_DISTANCE_THRESHOLD: 
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED) # Green click feedback
                    autopy.mouse.click()
                    # Add a small delay after click to prevent rapid multi-clicks
                    time.sleep(0.1) 

    # 7. Frame Rate Calculation
    curr_time = time.time()
    if prev_time > 0:
        fps = 1 / (curr_time - prev_time)
        cv2.putText(img, f"FPS: {int(fps)}", (20, 50), 
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    prev_time = curr_time

    # 8. Display Image
    cv2.imshow("Virtual Mouse", img)

    # 9. Exit Condition
    key = cv2.waitKey(1) & 0xFF 
    if key == ord('q'):
        print("Exiting...")
        break 

# --- Cleanup ---
print("Releasing camera and destroying windows...")
capture.release()
cv2.destroyAllWindows()
print("Done.")
