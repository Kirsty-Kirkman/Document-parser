"""
extract_shapes.py

Purpose:
Extract contours and lines from processed images.

Outputs structured geometry data.
"""
import cv2
import numpy as np
import json
import os

def extract_contours(processed_image: np.ndarray):
    """
    Extract contours (shapes) from processed image.

    Args:
        processed_image (np.ndarray): processed image

    Returns:
        list[dict]: list of dictionaries representing bounding boxes of shapes
    """
    # Contours represent connected regions
    contours, _ = cv2.findContours(
        image=processed_image,
        mode=cv2.RETR_EXTERNAL, # Only outer contours
        method=cv2.CHAIN_APPROX_SIMPLE # Compresses contour representation
    )

    shapes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        shapes.append({
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h)
        })
    return shapes

def extract_lines(edges_image: np.ndarray):
    """
    Extract lines from processed image.

    Args:
        edges_image (np.ndarray): processed image

    Returns:
        list[dict]: list of dictionaries representing lines
    """
    # Hough Transform for striaght line detection
    lines = cv2.HoughLinesP(
        image=edges_image,
        rho=1, # Distance resolution in pixels
        theta=np.pi / 180, # Angle resolution in radians
        threshold=100, # Minimum votes to detect line
        minLineLength=50, # Minimum line size
        maxLineGap=10 # Maximum allowed gap between segments to treat them as a single line
    )

    lines_list = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            lines_list.append({
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2)
            })
    return lines_list

def save_features(features: dict,
                  output_dir: str = "output"):
    """
    Save features as JSON.

    Args:
        features (dict): features extracted from image.
        output_dir (str, optional): output directory path. Defaults to "output".
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/features.json", "w") as f:
        json.dump(features, f, indent=4)

def extract_features(processed_image: np.ndarray,
                     edges_image: np.ndarray,
                     output_dir: str = "output"):
    """_summary_

    Args:
        processed_image (np.ndarray): processed image
        edges_image (np.ndarray): edge detected image
        output_dir (str, optional): output directory path. Defaults to "output".

    Returns:
        dict: extracted features (shapes and lines)
    """
    shapes = extract_contours(processed_image)
    lines = extract_lines(edges_image)
    features = {
        "shapes": shapes,
        "lines": lines
    }
    save_features(features, output_dir)
    return features