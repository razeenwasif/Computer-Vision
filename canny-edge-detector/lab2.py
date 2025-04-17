# COMP4528: COMPUTER VISION, S1 2025
# Lab 2: Implementing functions for building a Canny edge detector

import numpy as np
import cv2


def build_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Builds a 2D Gaussian kernel with the given size and standard deviation.
    Implement this by yourself, DO NOT use OpenCV.

    Parameters:
        size (int): The size of the square kernel (odd integer).
        sigma (float): The standard deviation of the Gaussian distribution.

    Returns:
        numpy.ndarray (size, size): A 2D Gaussian kernel as a square numpy array.
    """
    # size is the size of matrix like 3x3=9 or 5x5=25
    # formula: G(x,y) = 1/2πσ^2 ​e-(x2+y2/2σ^2)
    # lets say size = 3 (N). center coord would be 0,0
    # grid = [[(-1,-1) (-1,0) (-1,1)],
    #         [(0, -1) (0, 0) ( 0,1)],
    #         [(1, -1) (1, 0) (1, 1)]]
    # half size of kernel
    N = int(np.floor(size/2))
    # generate the matrix grid (if size = 3) N = 1
    # need -1 to N+1 (=1 since 0)
    x, y = np.meshgrid(np.arange(-N, N+1), np.arange(-N, N+1))
    # gaussian function 
    norm = 1 / (2 * np.pi * (sigma**2))
    gauss_kern = norm * np.exp(-(x**2 + y**2)/(2*(sigma**2)))
    # normalize kernel so it sums to 1
    gauss_kern = gauss_kern / np.sum(gauss_kern)
    return gauss_kern

def apply_filter(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Applies a 2D filter to a grayscale image with zero-padding using cross-correlation.
    Implement this by yourself, DO NOT use OpenCV.

    Parameters:
        image (numpy.ndarray (H, W)): The input grayscale image.
        kernel (numpy.ndarray (size, size)): The 2D Gaussian kernel.

    Returns:
        numpy.ndarray (H, W): The filtered image.
    """
    # dimensions
    img_h, img_w = img.shape 
    kern_h, kern_w = kernel.shape
    
    # padding 
    pad_h = int(np.floor(kern_h / 2))
    pad_w = int(np.floor(kern_w / 2))
    
    # dummy output
    rtn = np.zeros_like(img)

    # fill padded image with 0 making it bigger than the orig
    # img by pad_h and pad_w
    padded_img = np.zeros((img_h+2*pad_h, img_w+2*pad_w))
    # add zeros around orig img as padding
    padded_img[pad_h:pad_h+img_h, pad_w:pad_w+img_w] = img

    # apply correlation 
    for row in range(img_h):
        for col in range(img_w):
            overlap_region = padded_img[row:row+kern_h,
                                        col:col+kern_w]
            rtn[row, col] = np.sum(overlap_region*kernel)
    return rtn

def sobel_edge_detection(img: np.ndarray) -> tuple:
    """
    Applies Sobel edge detection to a grayscale image.
    Implement this by yourself, DO NOT use OpenCV.

    Parameters:
        image (numpy.ndarray (H, W)): The input grayscale image.

    Returns:
        tuple: (grad_x (H, W), grad_y (H, W)), the gradients in X and Y directions.
    """
    sobel_filter_h = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_filter_v = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

    # Get the dimensions of the image
    rows, cols = img.shape

    grad_x = np.zeros_like(img, dtype=np.float64)
    grad_y = np.zeros_like(img, dtype=np.float64)

    # Compute gradient using Sobel operators
    half_size = 3 // 2
    for i in range(half_size, rows - half_size):
        for j in range(half_size, cols - half_size):
            image = img[i - half_size:i + half_size + 1, j - half_size:j + half_size + 1]
            grad_x[i, j] = np.sum(image * sobel_filter_h)
            grad_y[i, j] = np.sum(image * sobel_filter_v)
    
    return (grad_x, grad_y)

def non_maximum_suppression(gradient_magnitude: np.ndarray, gradient_direction: np.ndarray) -> np.ndarray:
    """
    Applies Non-Maximum Suppression (NMS) along the gradient direction to a gradient magnitude image.

    Parameters:
        gradient_magnitude (numpy.ndarray (H, W)): The gradient magnitude.
        gradient_direction (numpy.ndarray (H, W)): The gradient direction (in degrees, 0-180).

    Returns:
        numpy.ndarray (H, W): The suppressed gradient magnitude.
    """
    # dims 
    rows, cols = gradient_magnitude.shape
    suppressed_magnitude = np.copy(gradient_magnitude)

    # find adjacent pixels in grad direction 
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            # angle 
            angle = gradient_direction[i][j]
            # get neighbours based on closest direction 
            # (0, 45, 90, 135)
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q0 = gradient_magnitude[i][j+1]
                q1 = gradient_magnitude[i][j-1]
            elif (22.5 <= angle < 67.5):
                q0 = gradient_magnitude[i-1][j+1]
                q1 = gradient_magnitude[i+1][j-1]
            elif (67.5 <= angle < 112.5):
                q0 = gradient_magnitude[i-1][j]
                q1 = gradient_magnitude[i+1][j]
            else: # 135
                q0 = gradient_magnitude[i+1][j+1]
                q1 = gradient_magnitude[i-1][j-1]

            if gradient_magnitude[i][j] < max(q0, q1):
                suppressed_magnitude[i][j] = 0
    return suppressed_magnitude

def double_thresholding(image: np.ndarray, low_threshold: float, high_threshold: float) -> np.ndarray:
    """
    Applies double thresholding to an edge image.
    For each pixel, if the gradient magnitude is higher than the high threshold,
    it is marked as a strong edge (255). If the gradient magnitude is between the low
    and high thresholds, it is marked as a weak edge (128). Otherwise, it is suppressed (0).

    Parameters:
        image (numpy.ndarray (H, W)): The gradient magnitude after NMS.
        low_threshold (float): Lower threshold for edge linking.
        high_threshold (float): Upper threshold for strong edges.

    Returns:
        numpy.ndarray (H, W): Binary edge map.
    """
    rows, cols = image.shape
    edge_map = np.zeros((rows, cols), dtype=np.uint8)

    strong_edge_i, strong_edge_j = np.where(image >= high_threshold)
    weak_edge_i, weak_edge_j = np.where((image >= low_threshold) & (image < high_threshold))
    
    # strong edge = white
    edge_map[strong_edge_i, strong_edge_j] = 255

    # weak edges
    for i, j in zip(weak_edge_i, weak_edge_j):
        if (edge_map[i-1:i+2][j-1:j+2] == 255).any():
            edge_map[i][j] = 255

    return edge_map

