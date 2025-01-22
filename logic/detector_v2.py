#!/usr/bin/env python3
import sys
from logic.line_clustering import group_by_dbscan
import cv2
import numpy as np
import imutils

def detector(input_image):
    """
    Applies the 'new algorithm' flow:
      1) Resize the image
      2) Gaussian blur
      3) Grayscale
      4) Morphological CLOSE
      5) Canny edge detection
      6) Another morph CLOSE
      7) HoughLinesP
      8)
    Returns:
      edges   - the post-processed edges
      output  - a blank image with the detected lines drawn
    """

    # (1) Resize to improve detection (scale factor ~1.6)
    scale_width = int(input_image.shape[1] * 1.6)
    resized = imutils.resize(input_image, width=scale_width)

    # (2) Gaussian blur
    kernel_size = 3
    blurred = cv2.GaussianBlur(input_image, (kernel_size, kernel_size), 0)

    # (3) Convert to grayscale
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # (4) Morphological CLOSE (to improve edges detection)
    kernel0 = np.ones((9, 27), np.uint8)
    closed_before_canny = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel0)

    # (5) Canny edge detection
    low_threshold = 5
    high_threshold = 50
    edges = cv2.Canny(closed_before_canny, low_threshold, high_threshold)

    # (6) Another morph CLOSE to merge edges
    kernel2 = np.ones((8, 24), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel2)

    # (7) HoughLinesP
    rho = 1
    theta = np.pi / 180
    threshold = 30
    min_line_length = 300
    max_line_gap = 20

    lines = cv2.HoughLinesP(
        closed_edges,
        rho,
        theta,
        threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    # Create a blank image (same size as the resized image) to draw lines on
    output = np.zeros_like(resized)

    line_list = []
    slope_list = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate the slope of the line
            slope = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            # Filter out purely vertical (slope ~90 or ~-90) and horizontal (slope ~0 or ~180) lines
            if not (slope == 0 or slope == 180 or slope == 90 or slope == -90):
                # Store line details for later processing
                line_list.append(((x1, y1), (x2, y2), slope))
                slope_list.append(slope)

    clusters = group_by_dbscan(slope_list)
    longest_lines = []
    for cluster in clusters:
        # Find the longest line in the cluster
        longest_line = None
        max_length = 0
        for line, slope in zip(line_list, slope_list):
            
            if slope in cluster:
                x1, y1 = line[0]
                x2, y2 = line[1]
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if length > max_length:
                    max_length = length
                    longest_line = line
        longest_lines.append(longest_line)
        # Draw the longest line in the cluster
        if longest_line:
            x1, y1= longest_line[0]
            x2, y2 = longest_line[1]
            thickness = 2
            cv2.line(output, (x1, y1), (x2, y2), (255, 255, 255), thickness)

    #temp return
    
    print(longest_lines[0])
    return (longest_lines[0])
    
    #return closed_edges, output


