#!/usr/bin/env python3
import sys
import cv2
import numpy as np
import imutils
from line_clustering import group_by_dbscan

def detector(input_image):
    """
    Refactored flow for detecting white lines on a soccer field:

      1) Resize image
      2) Extract field mask
      3) Canny edge detection
      4) HoughLinesP (probabalistic hough transform)
      5) Remove purely horizontal/vertical lines (so we can avoid false positives)
      6) Display intermediate steps (optional)
    """

    # (1) Resize image
    scale_factor = 1.6
    new_width = int(input_image.shape[1] * scale_factor)
    resized = imutils.resize(input_image, width=new_width)

    # (2) Extract field mask
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 10, 60])
    upper_green = np.array([65, 255, 255])
    field_mask = cv2.inRange(hsv, lower_green, upper_green)
    field_layer = cv2.bitwise_and(resized, resized, mask=field_mask)

    # (3) Canny edge detection
    kernel_size = 3
    blurred = cv2.GaussianBlur(resized, (kernel_size, kernel_size), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    # No more morphological close – directly run Canny on gray
    low_threshold, high_threshold = 5, 50
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    # (4) HoughLinesP on plain edges (no morphological close)
    rho, theta, threshold = 1, np.pi / 180, 30
    min_line_length, max_line_gap = 300, 20
    lines = cv2.HoughLinesP(
        edges,
        rho,
        theta,
        threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    # (5) Cluster lines by slope, then draw only the longest line per cluster
    output = resized.copy()
    line_list = []
    slope_list = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            # Skip purely horizontal/vertical lines
            if slope not in (0, 180, 90, -90):
                line_list.append(((x1, y1), (x2, y2), slope))
                slope_list.append(slope)

    if slope_list:
        clusters = group_by_dbscan(slope_list)
        for cluster in clusters:
            longest_line = None
            max_length = 0
            for (line_vals, slope_val) in zip(line_list, slope_list):
                if slope_val in cluster:
                    (pt1, pt2, _) = line_vals
                    x1, y1 = pt1
                    x2, y2 = pt2
                    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if length > max_length:
                        max_length = length
                        longest_line = line_vals

            if longest_line:
                x1, y1 = longest_line[0]
                x2, y2 = longest_line[1]
                cv2.line(output, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Now we just return edges (no morphological closed_edges)
    return resized, field_layer, edges, lines, output


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 detector_v2.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    original = cv2.imread(image_path)
    if original is None:
        print(f"Error: Could not open or find the image '{image_path}'.")
        sys.exit(1)

    # Adjusted return signature (dropped closed_edges)
    resized, field_layer, edges, lines, final_output = detector(original)

    # Show relevant outputs
    cv2.imshow("Edges", edges)
    cv2.imshow("Final Output (White Lines)", final_output)

    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
