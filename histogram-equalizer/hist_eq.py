# COMP4528: COMPUTER VISION, S1 2025
# Lab 1: Image Representation and Basic Processing

import cv2
import numpy as np


def load_image_and_convert_to_rgb(image_path: str) -> np.ndarray:
    """
    Load an image from the given file path and convert it to RGB format.

    Parameters:
        image_path (str): The path to the image file.

    Returns:
        numpy.ndarray: The RGB image as a 3D numpy array (H, W, 3).
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image from {image_path}")
        print(f"Image loaded from {image_path}")
        # convert to rgb 
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return rgb_image
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise


def resize_image(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Resize an image to the specified width and height.

    Parameters:
        img (numpy.ndarray): The input image as a 3D numpy array (H, W, 3).
        width (int): The desired width of the output image.
        height (int): The desired height of the output image.

    Returns:
        numpy.ndarray: The resized image as a 3D numpy array (height, width, 3).
    """
    if img is None:
        raise ValueError("Input image is None")
    return cv2.resize(img, (width, height))


def swap_red_and_blue_channels(img: np.ndarray) -> np.ndarray:
    """
    Swap the red and blue channels of an RGB image and return the new image.

    Parameters:
        img (numpy.ndarray): The input image as a 3D numpy array (H, W, 3).

    Returns:
        numpy.ndarray: The image with the red and blue channels swapped.
    """
    if img is None:
        raise ValueError("Input image is None")
    # [0,1,2] -> [R,G,B]
    return img[:, :, [2, 1, 0]]


def convert_to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to grayscale using the formula:
    gray = 0.299 * R + 0.587 * G + 0.114 * B

    Parameters:
        img (numpy.ndarray): The input image as a 3D numpy array (H, W, 3).

    Returns:
        numpy.ndarray: The grayscale image as a 2D numpy array (H, W).
    """
    if img is None:
        raise ValueError("Input image is None")
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    gray = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)
    return gray


def histogram_equalization(img: np.ndarray) -> np.ndarray:
    """
    Perform histogram equalization on each channel of a RGB image without using OpenCV's built-in functions.

    Parameters:
        img (numpy.ndarray): The input RGB image as a 3D numpy array (H, W, 3).

    Returns:
        numpy.ndarray: The equalized RGB image as a 3D numpy array (H, W, 3).
    """
    if img is None:
        raise ValueError("Input image is None")
    
    # Create an output image with the same shape as the input
    equalized_img = np.zeros_like(img)
    
    # Apply histogram equalization to each channel separately
    for i in range(3):
        channel = img[:, :, i]
        
        # Step 1: Calculate histogram
        hist = np.zeros(256, dtype=np.int32)
        for pixel_value in channel.flatten():
            hist[pixel_value] += 1
        
        # Step 2: Calculate cumulative distribution function (CDF)
        cdf = hist.cumsum()
        
        # Step 3: Normalize CDF to the range [0, 255]
        # Handle the case where min(cdf) might equal max(cdf)
        cdf_min = cdf[cdf > 0].min() if np.any(cdf > 0) else 0
        cdf_max = cdf.max()
        
        if cdf_max - cdf_min > 0:  # Avoid division by zero
            # Create a lookup table for the mapping
            lookup_table = np.zeros(256, dtype=np.uint8)
            for j in range(256):
                if cdf[j] > 0:
                    lookup_table[j] = np.round(((cdf[j] - cdf_min) * 255) / (cdf_max - cdf_min))
            
            # Step 4: Apply the mapping to the channel
            equalized_channel = lookup_table[channel]
            equalized_img[:, :, i] = equalized_channel
        else:
            # If all pixels have the same value, keep the original channel
            equalized_img[:, :, i] = channel
    
    return equalized_img


def main():
    """
    Example usage of the image processing functions.
    """
    # Path to the image file
    image_path = "./butterfly.jpg"
    
    try:
        # Load the image and convert to RGB
        rgb_image = load_image_and_convert_to_rgb(image_path)
        
        # Resize the image
        resized_image = resize_image(rgb_image, 400, 300)
        
        # Swap red and blue channels
        swapped_image = swap_red_and_blue_channels(rgb_image)
        
        # Convert to grayscale
        gray_image = convert_to_grayscale(rgb_image)
        
        # Apply histogram equalization
        equalized_image = histogram_equalization(rgb_image)
        
        # Display the results (optional)
        # Uncomment these lines if you want to display the images
        """
        cv2.imshow("Original (RGB)", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        cv2.imshow("Resized", cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))
        cv2.imshow("Swapped R-B", cv2.cvtColor(swapped_image, cv2.COLOR_RGB2BGR))
        cv2.imshow("Grayscale", gray_image)
        cv2.imshow("Histogram Equalized", cv2.cvtColor(equalized_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
        
        # Save the results
        cv2.imwrite("resized.jpg", cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite("swapped.jpg", cv2.cvtColor(swapped_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite("grayscale.jpg", gray_image)
        cv2.imwrite("equalized.jpg", cv2.cvtColor(equalized_image, cv2.COLOR_RGB2BGR))
        
        print("All operations completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main() 