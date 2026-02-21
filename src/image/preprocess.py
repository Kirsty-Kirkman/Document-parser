"""
preprocess.py

Purpose:
Load image and perform preprocessing to prepare for feature extraction.

Outputs cleaned images for downstream processing.
"""
from PIL import Image           # Used for loading image safely and format handling
import cv2                     # OpenCV for computer vision operations
import numpy as np            # Numerical operations and array handling
import os


def load_image(image_path: str) -> np.ndarray:
    """
    Load image from disk using PIL and convert to OpenCV format.

    Args:
        image_path: Path to input image

    Returns:
        OpenCV image (numpy array)
    """
    # Load using PIL first (better format support than cv2.imread)
    pil_image = Image.open(image_path).convert("RGB")

    # Convert PIL image → NumPy array → OpenCV format
    # PIL uses RGB
    # OpenCV uses BGR
    cv_image = cv2.cvtColor(
        np.array(pil_image),
        cv2.COLOR_RGB2BGR
    )
    return cv_image

def convert_to_grayscale(cv_image: np.ndarray) -> np.ndarray:
    """
    Convert color image to grayscale.

    Args:
        cv_image: input color image in OpenCV format

    Returns:
        np.ndarray: grayscale image (numpy array)
    """
    gray_image = cv2.cvtColor(
        cv_image,
        cv2.COLOR_BGR2GRAY
    )
    return gray_image

def remove_noise(gray_image: np.ndarray) -> np.ndarray:
    """
    Smoothen noise.

    Args:
        gray_image (np.ndarray): grayscale image

    Returns:
        np.ndarray: blurred image
    """
    # Gaussian blur smooths noise while preserving structure
    # Kernel size (5,5) = standard starting point
    # Larger kernel = more smoothing but more information loss
    blur_image = cv2.GaussianBlur(
        gray_image,
        ksize=(5, 5),
        sigmaX=0
    )
    # Alternative option:
    # blur = cv2.medianBlur(gray, 5)
    # Median blur is better for salt-and-pepper noise
    return blur_image

def threshold_image(blur_image: np.ndarray) -> np.ndarray:
    """
    Threshold image to binary (black and white).

    Args:
        blur_image (np.ndarray): blurred grayscale image

    Returns:
        np.ndarray: thresholded binary image
    """
    # This makes contour detection easier
    _, thresh_image = cv2.threshold(
        src=blur_image,
        thresh=150,
        maxval=255,
        type=cv2.THRESH_BINARY_INV
    )
    return thresh_image

def close_gaps(thresh_image: np.ndarray) -> np.ndarray:
    """
    Close small gaps in contours to connect broken lines.

    Args:
        thresh_image (np.ndarray): thresholded binary image

    Returns:
        np.ndarray: image with closed contours (small gaps filled)
    """
    # Closing fills small gaps and connects broken lines
    kernel = np.ones((3, 3), np.uint8)
    closed_image = cv2.morphologyEx(
        src=thresh_image,
        op=cv2.MORPH_CLOSE,
        kernel=kernel
    )
    return closed_image

def detect_edges(blur_image: np.ndarray) -> np.ndarray:
    """
    Edge detection.

    Args:
        blur_image (np.ndarray): thresholded binary image

    Returns:
        np.ndarray: edge-detected binary image
    """
    # Canny detects edges based on gradient change
    edges = cv2.Canny(
        image=blur_image,
        threshold1=50,
        threshold2=150
    )
    return edges

def save_intermediates(output_dir: str,
                       gray: np.ndarray,
                       blur: np.ndarray,
                       thresh: np.ndarray,
                       closing: np.ndarray,
                       edges: np.ndarray):

    os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(f"{output_dir}/gray.png", gray)
    cv2.imwrite(f"{output_dir}/blur.png", blur)
    cv2.imwrite(f"{output_dir}/threshold.png", thresh)
    cv2.imwrite(f"{output_dir}/processed.png", closing)
    cv2.imwrite(f"{output_dir}/edges.png", edges)

def preprocess_image(image_path: str,
                     output_dir: str = "output") -> dict:

    cv_image = load_image(image_path)
    gray = convert_to_grayscale(cv_image)
    blur = remove_noise(gray)
    thresh = threshold_image(blur)
    closing = close_gaps(thresh)
    edges = detect_edges(blur)
    save_intermediates(
        output_dir,
        gray,
        blur,
        thresh,
        closing,
        edges
    )

    return {
        "processed": closing,
        "edges": edges
    }