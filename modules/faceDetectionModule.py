import cv2 
import mediapipe as mp 
import time

# --- Configuration --- 
videoPath = "./testAssets/3192305-uhd_3840_2160_25fps.mp4" 

# --- Video capture ---
capture = cv2.VideoCapture(videoPath)

if not capture.isOpened():
    print(f"Error: Could not open video file: {video_path}")
    exit()

# --- get video properties --- 
orig_fps = capture.get(cv2.CAP_PROP_FPS)
if orig_fps == 0: # cases where fps might not be available
    print("Warning: Could not get video FPS. Using default delay.")
    wait_time = 30
else:
    wait_time = int(1000 / orig_fps)
    wait_time = max(1, wait_time)
print(f"Original Video FPS: {orig_fps:.2f}, Wait time per frame: {wait_time} ms")

# --- mediapipe face detection setup ---
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
min_detection_confidence = 0.5
faceDetection = mpFaceDetection.FaceDetection(
    min_detection_confidence = min_detection_confidence
)

prevTime = 0

while True:
    success, img = capture.read()

    # --- exit if end of video --- 
    if not success:
        print("End of vid reached or failed to read frame")
        break 

    # --- process frame --- 
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)

    # --- draw detections --- 
    if results.detections:
        for id, detection in enumerate(results.detections):
            # basic bounding box and keypoints
            mpDraw.draw_detection(img, detection)
            score = detection.score
            print(f"Probability this is face: {score}")

    # --- calculate and display processing fps
    currTime = time.time()
    if prevTime > 0:
        fps = 1 / (currTime - prevTime)
        
        cv2.putText(img, f"FPS: {int(fps)}", (20, 70), 
                    cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 2)
    
    prevTime = currTime

    # --- display image ---
    cv2.imshow("Face Detection", img)
    
    key = cv2.waitKey(wait_time) & 0xFF
    if key == ord('q'):
        print("exiting")
        break 

# --- Release resources --- 
print("Releasing video capture...")
capture.release()
print("Destroying all windows...")
cv2.destroyAllWindows()
print("Done.")


