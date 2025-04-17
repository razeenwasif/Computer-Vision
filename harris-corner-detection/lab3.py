# COMP4528: COMPUTER VISION, S1 2025
# Lab 3: Image Feature Detection and Description

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def compute_harris_response(Ix: np.ndarray, Iy: np.ndarray, k: float=0.05,
    sigma: float=1.0) ->np.ndarray:
    """Computes the Harris corner response map using the gradient images Ix and Iy.
    You may use the imported gaussian_filter function from scipy.

    Parameters:
        Ix (numpy.ndarray (H, W) [float]): The gradient image in the x-direction.
        Iy (numpy.ndarray (H, W) [float]): The gradient image in the y-direction.
        k (float): Harris detector free parameter, typically between 0.01 and 0.1. Defaults to 0.05.
        sigma (float): Standard deviation for Gaussian filter. Defaults to 1.

    Returns:
        numpy.ndarray (H, W) [float]: The Harris response map.
    """
    # product of derivatives at each pixel
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy 
    Ixy = Ix * Iy 

    # use gaussian filtering to smooth the products
    Ix2_smooth = gaussian_filter(Ix2, sigma=sigma)
    Iy2_smooth = gaussian_filter(Iy2, sigma=sigma)
    Ixy_smooth = gaussian_filter(Ixy, sigma=sigma)

    # Compute the harris response function 
    # R = det(M) - k * (trace(M))**2
    # M is the structure tensor [ [Ix2, Ixy], [Ixy, Iy2] ]

    # Determinant = Ix2 * Iy2 - Ixy**2 
    det_M = Ix2_smooth * Iy2_smooth - (Ixy_smooth * Ixy_smooth)
    # trace = Ix2 + Iy2 
    trace_M = Ix2_smooth + Iy2_smooth

    # Harris response
    R = det_M - k * (trace_M**2)

    return R

def harris_keypoint_nms(response: np.ndarray, threshold: float=0.01,
    window_size: int=5) ->list:
    """Identifies keypoints based on the Harris response using non-maximum suppression.

    Args:
        response (np.ndarray (H, W) [float]): The Harris response map.
        threshold (float, optional): Threshold for keypoint detection. Defaults to 0.01.
        window_size (int, optional): Side-length (odd) of the square window for non-maximum suppression. Defaults to 5.

    Returns:
        list: A list of identified keypoints as (x, y) integer coordinates.
    """
    # ensure window size is odd:
    assert window_size % 2 == 1, "Window size must be odd"

    # height and width of the response map
    height, width = response.shape

    # get half of window for neighbourhood checking
    half_window = window_size // 2

    keypoints = []

    # iterate through all pixels in the response map (excl. border)
    for y in range(half_window, height - half_window):
        for x in range(half_window, width - half_window):
            # skip if response < threshold
            if response[y, x] < threshold:
                continue 
            window = response[y - half_window:y + half_window + 1,
                            x - half_window:x + half_window + 1]
            # check if center pixel is the maximum in the window 
            if response[y, x] == np.max(window):
                # add as keypoint 
                keypoints.append((x, y))
    return keypoints


def compute_DoG(image: np.ndarray, sigma: float=1.6, num_octaves: int=4,
    num_scales: int=5) ->list[list[np.ndarray]]:
    """Computes the Difference of Gaussians (DoG) pyramid for the given image.

    Args:
        image (np.ndarray (H, W) [uint8]): The input grayscale image with shape (H, W).
        sigma (float, optional): The base standard deviation of the Gaussian filter for the first octave. Defaults
            to 1.6, as suggested in SIFT.
        num_octaves (int, optional): Number of octaves in the pyramid. Each octave represents an image downsampled by a
            factor of 2. Defaults to 4.
        num_scales (int, optional): Number of scales per octave (excluding extra layers for DoG computation). Defaults
            to 5.

    Returns:
        list[list[np.ndarray]]: A DoG pyramid, where each inner list represents an octave and contains `num_scales` DoG
            images. Each DoG image has the same shape as the corresponding Gaussian-blurred image in that octave.
            That is, if images in the first octave have shape (H, W) [float32], then the second octave images have
            shape (H//2, W//2) [float32].

    Notes:
        1. The input image should be converted to floating point and **normalized to the range [0,1]** by dividing by 255.0
            to ensure numerical stability.
        2. The standard deviation for each Gaussian blur is computed as:
            sigma_i = sigma * (k^i), where k = 2^(1/num_scales) and i is the scale index.
        3. Since `num_scales + 1` Gaussian images are generated per octave, we get `num_scales` DoG images.
        4. Each **octave** starts from an image and iteratively applies Gaussian smoothing.
           Once all scales are computed, the image is downsampled (reduced to half resolution)
           and used as the base for the next octave.
        5. Use (0, 0) as the kernel size for cv2.GaussianBlur to automatically compute the kernel size.
    """
    img = image.astype(np.float32) / 255.0
    # scale factor
    k = 2.0 ** (1.0 / num_scales)

    dog_pyramid = []
    # curr base img for the octave 
    curr_base_img = img.copy()

    for octave in range(num_octaves):
        # store gaussian blurred imgs for the octave
        gauss_imgs = []
        # store DoG images for this octave 
        dog_imgs = []

        # generate num_scales + 1 gaussian-blurred images for the octave
        for scale in range(num_scales + 1):
            scale_sigma = sigma * (k**scale)
            blurred = cv2.GaussianBlur(curr_base_img, (0,0), scale_sigma)
            gauss_imgs.append(blurred)

            # compute DoG as difference b/w consecutive Gauss imgs 
            if scale > 0:
                DoG = gauss_imgs[scale] - gauss_imgs[scale - 1]
                dog_imgs.append(DoG)
        
        # add the DoG images for this octave to the pyramid 
        dog_pyramid.append(dog_imgs)

        # prepare base img for next octave by downsampling the last gaussian
        # image of the current octave
        if octave < num_octaves - 1:
            curr_base_img = cv2.resize(gauss_imgs[-1],
                                    (gauss_imgs[-1].shape[1] // 2,
                                    gauss_imgs[-1].shape[0] // 2),
                                    interpolation=cv2.INTER_AREA)
    return dog_pyramid


def match_features(descriptors1: np.ndarray, descriptors2: np.ndarray,
    ratio_thresh: float=0.75) ->list:
    """Matches features between two descriptor sets using the L2 distance (np.linalg.norm).
    A feature in the first set (descriptors1) is matched to the closest feature in the second set (descriptors2),
    in the L2 sense, so long as it passes Lowe's Ratio Test.

    Args:
        descriptors1 (np.ndarray (N, 128) [float32]): Descriptors from image 1.
        descriptors2 (np.ndarray (N, 128) [float32]): Descriptors from image 2.
        ratio_thresh (float): Lowe’s Ratio Test threshold.

    Returns:
        matches (list[tuple]): Matched keypoint indices [(idx1_0, idx2_0), ..., (idx1_k, idx2_k)].
    """
    matches = []
    # for each descriptor in the first set, init variables to track the 
    # two closest matches 
    for i, desc1 in enumerate(descriptors1):
        best_dist = float('inf')
        second_best_dist = float('inf')
        best_idx = -1

        # compare with all desc in the second set
        for j, desc2 in enumerate(descriptors2):
            # compute L2 dist b/w descriptors 
            dist = np.linalg.norm(desc1 - desc2)

            if dist < best_dist:
                second_best_dist = best_dist
                best_dist = dist
                best_idx = j 
            elif dist < second_best_dist:
                second_best_dist = dist 
        
        # lowe's ratio test 
        if best_dist < ratio_thresh * second_best_dist:
            matches.append((i, best_idx))
    
    return matches


