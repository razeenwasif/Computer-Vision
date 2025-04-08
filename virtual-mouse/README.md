# Virtual Mouse using Hand Tracking :mouse::raised_hand:

This project allows you to control your computer's mouse cursor using hand gestures detected through your webcam. It utilizes OpenCV for video capture, MediaPipe for accurate hand landmark detection, and Autopy for controlling the mouse.

## Features

*   **Cursor Movement:** Control the mouse pointer by moving your index finger.
*   **Clicking:** Perform a left mouse click by bringing your index and middle fingertips close together.
*   **Real-time Visual Feedback:** See your hand landmarks and active control area overlaid on the webcam feed.
*   **Adjustable Smoothing:** Implements the One-Euro Filter for smoother, less jittery cursor movement, with tunable parameters.
*   **Modular Design:** Hand tracking and filtering logic are separated into reusable modules.

## How it Works

1.  **Video Capture:** OpenCV captures frames from the default webcam.
2.  **Hand Detection:** MediaPipe (via `modules/handTrackingModule.py`) processes each frame to detect hand landmarks in real-time.
3.  **Gesture Recognition:** The application checks which fingers are raised (`fingersUp` method in the hand tracking module).
    *   **Index Finger Up:** Enters "Moving Mode".
    *   **Index & Middle Fingers Up:** Enters "Clicking Mode".
4.  **Coordinate Mapping:** The position of the index fingertip (within a defined region of the camera frame) is mapped to the computer screen's coordinates using `numpy.interp`.
5.  **Smoothing:** The mapped coordinates are passed through a One-Euro Filter (from `modules/filters.py`) to reduce jitter and lag, providing smoother cursor control.
6.  **Mouse Control:** Autopy library is used to:
    *   Move the mouse cursor to the smoothed coordinates (`autopy.mouse.move`).
    *   Perform a left click when the distance between index and middle fingertips is below a threshold (`autopy.mouse.click`).
7.  **Display:** An OpenCV window shows the webcam feed with landmarks, active area, and visual feedback for current actions.


## Requirements

*   Python 3.7+
*   OpenCV (`opencv-python`)
*   MediaPipe (`mediapipe`)
*   Autopy (`autopy`)
*   NumPy (`numpy`)

## Installation

1.  **Clone the Repository (or Download Files):**
    ```bash
    git clone <your-repository-url> # Or download and extract the ZIP
    cd your_project_folder
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment:**
    *   **Windows:** `.\venv\Scripts\activate`
    *   **macOS/Linux:** `source venv/bin/activate`

4.  **Create `requirements.txt`:**
    Create a file named `requirements.txt` in the `your_project_folder/` directory with the following content:
    ```txt
    opencv-python
    mediapipe
    autopy
    numpy
    ```

5.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Autopy installation can sometimes have platform-specific issues. Refer to the [official Autopy documentation](https://github.com/autopilot-rs/autopy-py) if you encounter problems.)*

## Usage

1.  **Ensure your webcam is connected and unobstructed.**
2.  **Navigate to the project's root directory** (`your_project_folder/`) in your terminal.
3.  **Activate the virtual environment** (if you created one):
    *   Windows: `.\venv\Scripts\activate`
    *   macOS/Linux: `source venv/bin/activate`
4.  **Run the main application script:**
    ```bash
    python virtual-mouse/my_hand_app.py
    ```
5.  **Control the Mouse:**
    *   **Move:** Raise only your **index finger**. Move your hand within the frame. The cursor will follow your index fingertip's position (relative to the active area shown by the rectangle).
    *   **Click:** Raise your **index and middle fingers**. Bring the tips of these two fingers close together (within the `CLICK_DISTANCE_THRESHOLD`). A green circle will appear briefly at the midpoint when a click is registered.
    *   Lower your fingers or move them apart to stop clicking.
6.  **Exit:** Press the 'q' key while the display window is active.

## Configuration & Tuning

You can adjust the behavior by modifying constants in `virtual-mouse/my_hand_app.py`:

*   **`FRAME_REDUCTION`:** Defines the padding around the camera frame used as the active area for mapping hand position to the screen. Increasing this makes the active area smaller.
*   **`CLICK_DISTANCE_THRESHOLD`:** The maximum distance (in pixels on the camera feed) between the index and middle fingertips to trigger a click. Decrease for more sensitivity (easier to click), increase if accidental clicks occur.
*   **One-Euro Filter Parameters:**
    *   **`filter_min_cutoff`:** Controls smoothness at low speeds. Lower values = more smoothing (less jitter when still), higher values = less smoothing (more responsive to small movements). (Default: `1.0`)
    *   **`filter_beta`:** Controls how quickly smoothing decreases as speed increases. Higher values = less lag during fast movements, lower values = more consistent smoothing (potentially more lag). (Default: `0.007`)
    *   **`filter_d_cutoff`:** Cutoff for filtering the speed calculation itself. Usually fine at `1.0`.

Experiment with `filter_min_cutoff` and `filter_beta` to find the cursor feel that works best for you.

## Limitations & Potential Issues

*   Requires good and consistent lighting conditions for reliable hand tracking.
*   The hand must be clearly visible within the camera frame.
*   Performance depends on your computer's CPU/GPU capabilities. Low FPS can make control difficult.
*   Accidental clicks or movements can occur, especially during fast transitions or if tracking is momentarily lost.
*   Currently only implements move and left-click functionalities.
*   The coordinate mapping (`screen_width - smooth_x`) assumes the webcam image is *not* horizontally flipped. If you add `cv2.flip(img, 1)`, you need to remove the `screen_width - ` part in `autopy.mouse.move`.

## Future Improvements

*   Implement scrolling (e.g., using thumb + index finger distance, or number of fingers up).
*   Implement drag-and-drop (e.g., holding the "click" gesture).
*   Add right-click functionality (e.g., using ring finger + middle finger?).
*   Create a simple GUI for easier configuration of parameters.
*   Improve robustness to different lighting conditions or temporary tracking loss.
