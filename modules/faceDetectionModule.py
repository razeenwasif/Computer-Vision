import cv2
import mediapipe as mp
import time

class FaceDetector():
    """
    A class to detect faces in images using the MediaPipe Face Detection model.

    Attributes:
        min_detection_confidence (float): Minimum confidence value ([0.0, 1.0]) 
            for face detection to be considered successful.

        model_selection (int): 0 or 1. 0 to select a short-range model that works
            best for faces within 2 meters from the camera, and 1 for
            a full-range model best for faces within 5 meters.

        static_image_mode (bool): Whether to treat the input images as a batch of 
            static and possibly unrelated images, or a video stream.
            (Though Face Detection itself doesn't use this like Pose/Hands)
                                   
    Methods:
        findFaces(img, draw=True): Finds faces in an image and optionally draws results.
        fancyDraw(img, bbox, score, l=30, t=5, rt=1): Draws a stylized bounding box.
    """
    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        """
        Initializes the FaceDetector object.

        Args:
            min_detection_confidence (float): Minimum detection confidence threshold.
            model_selection (int): 0 for short-range, 1 for full-range model.
        """
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection

        # Initialize MediaPipe Face Detection components
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(
            self.min_detection_confidence, self.model_selection
        )
        self.results = None # To store the latest detection results

    def findFaces(self, img, draw=True):
        """
        Detects faces in a BGR image.

        Args:
            img (numpy.ndarray): The input image in BGR format.
            draw (bool): Whether to draw the bounding boxes and scores on the image.

        Returns:
            tuple: A tuple containing:
                - img (numpy.ndarray): The image with detections drawn (if draw=True).
                - bboxes (list): A list of dictionaries, where each dictionary 
                                 contains 'id', 'bbox', and 'score' for a detected face.
                                 'bbox' is [x, y, w, h].
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxes = []

        h, w, c = img.shape

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # --- Extract score and bounding box ---
                score = detection.score[0] 
                bboxC = detection.location_data.relative_bounding_box
                
                if bboxC is None:
                    continue 
                    
                # Convert relative coordinates to pixel coordinates
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                       int(bboxC.width * w), int(bboxC.height * h)
                
                bboxes.append({"id": id, "bbox": bbox, "score": score})

                # --- Draw on the original image (img) ---
                if draw:
                    # built-in simple drawing
                    # self.mpDraw.draw_detection(img, detection) 
                    
                    # custom drawing
                    img = self.fancyDraw(img, bbox, score) 
                    #cv2.rectangle(img, bbox, (255, 0, 255), 2) # Simple rectangle
                    #cv2.putText(img, f"{int(score * 100)}%",
                    #             (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                    #             2, (255, 0, 255), 2)


        return img, bboxes

    def fancyDraw(self, img, bbox, score, l=30, t=5, rt=1):
        """
        Draws a stylized bounding box with confidence score.

        Args:
            img (numpy.ndarray): Image to draw on.
            bbox (tuple): Bounding box coordinates (x, y, w, h).
            score (float): Confidence score of the detection.
            l (int): Length of the corner lines.
            t (int): Thickness of the lines.
            rt (int): Thickness of the rectangle outline.

        Returns:
            numpy.ndarray: Image with the fancy drawing.
        """
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        color = (255, 0, 255)

        # Draw main rectangle outline
        cv2.rectangle(img, bbox, color, rt)

        # Draw corner lines
        # Top Left x,y
        cv2.line(img, (x, y), (x + l, y), color, t)
        cv2.line(img, (x, y), (x, y + l), color, t)
        # Top Right x1,y
        cv2.line(img, (x1, y), (x1 - l, y), color, t)
        cv2.line(img, (x1, y), (x1, y + l), color, t)
        # Bottom Left x,y1
        cv2.line(img, (x, y1), (x + l, y1), color, t)
        cv2.line(img, (x, y1), (x, y1 - l), color, t)
        # Bottom Right x1,y1
        cv2.line(img, (x1, y1), (x1 - l, y1), color, t)
        cv2.line(img, (x1, y1), (x1, y1 - l), color, t)
        
        # --- Add Confidence Score Text ---
        text_x = bbox[0]
        text_y = bbox[1] - 10 
        if text_y < 10: # If too close to the top, put it inside
             text_y = bbox[1] + 20 
             
        cv2.putText(img, f"{int(score * 100)}%",
                    (text_x, text_y), cv2.FONT_HERSHEY_PLAIN,
                    2, color, 2)

        return img

# --- Usage ---
def main():
    # --- Configuration ---
    # videoPath = 0 # webcam
    videoPath = "./testAssets/855564-hd_1920_1080_24fps.mp4"

    # --- Video capture ---
    capture = cv2.VideoCapture(videoPath)
    if not capture.isOpened():
        print(f"Error: Could not open video source: {videoPath}")
        exit()

    # --- Determine wait time ---
    orig_fps = capture.get(cv2.CAP_PROP_FPS)
    if videoPath == 0 or orig_fps == 0: # Webcam or unavailable FPS
        wait_time = 1 # Process as fast as possible for webcam
        print("Using webcam or could not get video FPS. Setting wait_time=1ms.")
    else:
        wait_time = int(1000 / orig_fps)
        wait_time = max(1, wait_time)
        print(f"Original Video FPS: {orig_fps:.2f}, Wait time per frame: {wait_time} ms")

    # --- Initialize Detector ---
    detector = FaceDetector(min_detection_confidence=0.75)
    
    prevTime = 0

    while True:
        success, img = capture.read()
        if not success:
            print("End of video or failed to read frame.")
            break
        
        if videoPath == 0: # Flip webcam image for mirror view
            img = cv2.flip(img, 1)

        # --- Use Detector ---
        img, bboxes = detector.findFaces(img, draw=True) 

        if bboxes:
            # print the center of the first detected face
            first_bbox = bboxes[0]['bbox']
            cx = first_bbox[0] + first_bbox[2] // 2
            cy = first_bbox[1] + first_bbox[3] // 2
            print(f"Center of first face: ({cx}, {cy}), Score: {bboxes[0]['score']:.2f}")
            pass # No extra printing needed as score is drawn

        # --- Calculate and Display Processing FPS ---
        currTime = time.time()
        if prevTime > 0:
            fps = 1 / (currTime - prevTime)
            cv2.putText(img, f"FPS: {int(fps)}", (20, 70),
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        prevTime = currTime

        # --- Display image ---
        cv2.imshow("Face Detection", img)

        # --- Wait and check for exit key ---
        key = cv2.waitKey(wait_time) & 0xFF
        if key == ord('q'):
            print("Exiting...")
            break

    # --- Release resources ---
    print("Releasing video capture...")
    capture.release()
    print("Destroying all windows...")
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
