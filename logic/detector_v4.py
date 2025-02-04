#!/usr/bin/env python3

import sys
import cv2
import numpy as np
from logic.line_clustering import cluster_lines


def detector_v4(input_image):
    """
    Similar to detector_v2, this function accepts an input image and returns
    the list of hough_lines that are neither purely vertical nor purely horizontal.

    :param input_image: A BGR image (NumPy array) read by OpenCV.
    :return: A list of lines (each line is [x1, y1, x2, y2]) that passed the slope filter.
    """

    # Convert from BGR to HSV
    hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

    # Create a green mask (adjust lower_green and upper_green for your lighting)
    lower_green = (36, 25, 25)
    upper_green = (86, 255, 255)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Keep only the green field
    frame_masked = cv2.bitwise_and(input_image, input_image, mask=mask_green)

    # Convert to grayscale
    gray = cv2.cvtColor(frame_masked, cv2.COLOR_BGR2GRAY)

    # Optional simple binary threshold
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Canny edge detection
    canny = cv2.Canny(thresh, 50, 150, apertureSize=3)

    # Apply Hough Line Detection (Probabilistic)
    lines = cv2.HoughLinesP(
        canny,
        1,                # rho resolution
        np.pi / 180,      # theta resolution
        50,               # threshold
        minLineLength=150,
        maxLineGap=20
    )

    # Filter out purely vertical / horizontal lines
    hough_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Check for purely vertical lines
            if x2 - x1 == 0:
                # Purely vertical — skip
                continue

            slope = (y2 - y1) / (x2 - x1)
            # Skip purely horizontal (slope == 0)
            if slope != 0:
                print(x1, y1, x2, y2, slope)
                hough_lines.append([x1, y1, x2, y2, slope])

    clusters = cluster_lines(hough_lines)
    print(clusters)
    print(hough_lines)
    return hough_lines


