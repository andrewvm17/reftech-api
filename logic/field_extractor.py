#!/usr/bin/env python3

import cv2
import sys
import numpy as np

def extract_field_mask(image_bgr, hue_range=10, morph_size=15):
    """
    Extract a binary mask of the soccer field from a BGR input image.
    
    Parameters:
    -----------
    image_bgr  : np.ndarray
        Input image in BGR format (as loaded by cv2.imread).
    hue_range  : int
        Half-width of the hue threshold around the peak hue.
    morph_size : int
        Size (in pixels) for morphological opening/closing.
    
    Returns:
    --------
    field_mask : np.ndarray
        A binary (single-channel) mask, where pixels belonging to the field are 255, others 0.
    """
    #cv2.imshow("original image", image_bgr)
    # 1. Convert to HSV
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Split out the H, S, V channels
    hue_channel, _, _ = cv2.split(image_hsv)
    
    # 2. Compute a histogram of hue values (0..179 in OpenCV by default)
    hist_size = 180
    hist_range = (0, 180)
    hue_hist = cv2.calcHist([hue_channel], [0], None, [hist_size], hist_range)
    
    # Find the peak hue value in the histogram
    peak_hue = np.argmax(hue_hist)
    
    # 3. Threshold around the peak in the hue channel
    lower_hue = max(0, peak_hue - hue_range)
    upper_hue = min(179, peak_hue + hue_range)
    lower_bound = np.array([lower_hue, 0, 0], dtype=np.uint8)
    upper_bound = np.array([upper_hue, 255, 255], dtype=np.uint8)
    intermediate_mask = cv2.inRange(image_hsv, lower_bound, upper_bound)
    
    # 4. Morphological opening and closing to remove spurious regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_size, morph_size))
    # Opening removes small objects
    mask_opened = cv2.morphologyEx(intermediate_mask, cv2.MORPH_OPEN, kernel)
    # Closing fills small holes
    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)

    # 5. Find contours and keep the largest one as the field
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # No contours found: return the closed mask as-is
        return mask_closed
    
    # Sort by area, pick the largest
    largest_contour = max(contours, key=cv2.contourArea)

    # Contour approximation
    epsilon = 0.01 * cv2.arcLength(largest_contour, True)
    approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Convex hull
    hull = cv2.convexHull(approx_contour)
    
    # Create a new mask for the final field region
    field_mask = np.zeros_like(mask_closed)
    cv2.fillConvexPoly(field_mask, hull, 255)

    return field_mask


