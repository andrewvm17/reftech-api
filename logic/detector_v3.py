#!/usr/bin/env python3

"""
detector_v3.py

Script to detect lines on a soccer field, segment them by angle via k-means,
and find intersection points (e.g., corners of the lines).
Usage:
    python3 detector_v3.py <image_path>
"""

import sys
import math
import numpy as np
import cv2
from collections import defaultdict

def GetFieldLayer(src_img):
    """
    Extracts the 'field' layer by masking out green areas in the image.
    Returns an image where only the field's green regions are retained.
    """
    hsv = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV)
    # Define green range (tweak as needed)
    lower_green = np.array([35, 10, 60])
    upper_green = np.array([65, 255, 255])
    # Create mask for field
    field_mask = cv2.inRange(hsv, lower_green, upper_green)
    # Bitwise-and for the field layer
    field_layer = cv2.bitwise_and(src_img, src_img, mask=field_mask)
    return field_layer

def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """
    Group lines by their angle using k-means clustering.
    Typically used to separate 'vertical' from 'horizontal' lines.
    """

    # Define default criteria (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # Extract angles in [0, pi]
    angles = np.array([line[0][1] for line in lines], dtype=np.float32)

    # Multiply angles by 2, find points on the unit circle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)] for angle in angles], dtype=np.float32)

    # Run k-means (Python 3.x)
    _, labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)
    labels = labels.reshape(-1)  # Flatten

    # Segment lines by label
    segmented = defaultdict(list)
    for i, line in enumerate(lines):
        segmented[labels[i]].append(line)

    segmented = list(segmented.values())
    print(f"Segmented lines into two groups: {len(segmented[0])}, {len(segmented[1])}")

    return segmented

def intersection(line1, line2):
    """
    Find the intersection of two lines given in Hesse normal form.
    Returns the (x, y) coordinates as integers.
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]

def segmented_intersections(segmented_lines):
    """
    Find intersections between groups of lines:
    e.g., intersection of each line in group[i] with each line in group[i+1].
    """
    intersections = []
    for i, group in enumerate(segmented_lines[:-1]):
        for next_group in segmented_lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2))
    return intersections

def drawLines(img, lines, color=(0,0,255), thickness=2):
    """
    Draw Hough lines on the image in the specified color.
    Each line is (rho, theta) in Hesse normal form.
    """
    for line in lines:
        for (rho, theta) in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(img, (x1,y1), (x2,y2), color, thickness)

def detect_lines_and_intersections(src_img):
    """
    Main workflow:
      1. Extract the field layer (masking green).
      2. Detect edges, run HoughLines.
      3. Segment lines by angle.
      4. Find intersections between angle groups.
      5. Draw lines and intersections on a copy of the original image.

    Returns the final annotated image.
    """
    # 1) Extract field layer
    field_img = GetFieldLayer(src_img)

    # 2) Edge detection
    edges = cv2.Canny(field_img, 50, 200)

    # 3) HoughLines
    # Adjust parameters as needed
    rho = 3
    theta = np.pi / 50
    thresh = 400
    lines = cv2.HoughLines(edges, rho, theta, thresh)

    if lines is None or len(lines) == 0:
        print("No lines detected.")
        return src_img  # Return original if no lines found

    print(f"Found lines: {len(lines)}")

    # 4) Segment lines by angle (k=2 => typically vertical & horizontal)
    segmented = segment_by_angle_kmeans(lines, k=2)

    # 5) Find intersections
    intersections = segmented_intersections(segmented)
    print(f"Found total intersections: {len(intersections)}")

    # 6) Draw everything
    # Make a copy of the original for final visualization
    output_img = np.copy(src_img)

    # a) Draw one cluster (say vertical) in green
    #vertical_lines = segmented[1] if len(segmented) > 1 else []
    #drawLines(output_img, vertical_lines, color=(0,255,0))

    # b) Draw the other cluster (horizontal) in yellow
    horizontal_lines = segmented[0] if len(segmented) > 0 else []
    drawLines(output_img, horizontal_lines, color=(0,255,255))

    # c) Draw intersection points in magenta
    #for point in intersections:
     #   pt = (point[0][0], point[0][1])
      #  cv2.circle(output_img, pt, 5, (255, 0, 255), -1)

    return output_img

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 detector_v3.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    src_img = cv2.imread(image_path)

    if src_img is None:
        print(f"Error: Could not open or find the image '{image_path}'.")
        sys.exit(1)

    # Perform line + intersection detection
    result_img = detect_lines_and_intersections(src_img)

    # Show the final annotated image
    cv2.imshow("Original", src_img)
    cv2.imshow("Annotated", result_img)
    print("Press any key on the image window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
