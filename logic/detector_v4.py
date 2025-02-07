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
   # cv2.imshow("HSV", hsv)
    # Create a green mask (adjust lower_green and upper_green for your lighting)
    lower_green = (36, 25, 5)
    upper_green = (86, 255, 255)

    lower_white = (0, 0, 85)
    upper_white = (85,75, 255)    

    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    
    # Keep only the green field
    frame_masked_green = cv2.bitwise_and(input_image, input_image, mask=mask_green)
    #cv2.imshow("Frame Masked Green", frame_masked_green)
   # frame_masked_white = cv2.bitwise_and(frame_masked_green, frame_masked_green, mask=mask_white)
    #cv2.imshow("Frame Masked White", frame_masked_white)
    # Convert to grayscale
    gray = cv2.cvtColor(frame_masked_green, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Gray", gray)
    # Optional simple binary threshold
    #_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    #cv2.imshow("Thresh", thresh)
    # Canny edge detection
    canny = cv2.Canny(gray, 1, 150, apertureSize=3, L2gradient=True)
    #cv2.imshow("Canny", canny)
    # Apply Hough Line Detection (Probabilistic)
    lines = cv2.HoughLinesP(
        canny,
        1,                # rho resolution
        np.pi / 180,      # theta resolution
        150,               # threshold
        minLineLength=150,
        maxLineGap=25
    )
    #cv2.imshow("Hough Lines", lines)
    # Filter out purely vertical / horizontal lines, then keep only positive-slope (red) lines
    red_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Skip purely vertical lines
            if x2 - x1 == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)
            # Skip slope == 0 as well
            if slope != 0:
                # Keep only positive slopes for "red" lines
                if slope > 0:
                    # Compute line length
                    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    red_lines.append([x1, y1, x2, y2, slope, length])

    # Sort red_lines by length (desc), then keep only the top two
    red_lines.sort(key=lambda ln: ln[5], reverse=True)
    longest_red_lines = red_lines[:2]

    # Return only the two longest "red" lines
    return longest_red_lines


def line_segment_intersection(l1, l2):
    """
    Return the (x, y) intersection of two line segments (if it exists),
    otherwise return None.

    Each line is [x1, y1, x2, y2, m].
    """
    x1, y1, x2, y2, m1 = l1
    x3, y3, x4, y4, m2 = l2

    # Convert each segment to general line form: A*x + B*y = C
    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1

    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3

    determinant = A1 * B2 - A2 * B1

    # If determinant is 0, lines are parallel or coincident
    if determinant == 0:
        return None

    # Intersection point of the infinite lines
    x_int = (B2 * C1 - B1 * C2) / determinant
    y_int = (A1 * C2 - A2 * C1) / determinant

    # Check if within segment bounds for both lines
    if (min(x1, x2) <= x_int <= max(x1, x2) and
        min(y1, y2) <= y_int <= max(y1, y2) and
        min(x3, x4) <= x_int <= max(x3, x4) and
        min(y3, y4) <= y_int <= max(y3, y4)):
        return (x_int, y_int)

    return None

'''
def main():
    """
    Command-line entry point so you can run:
        python3 detector_v4.py offside.png
    This will run detector_v4() on the supplied image, 
    display the detected lines, and print their intersections.
    """
    if len(sys.argv) < 2:
        print("Usage: python3 detector_v4.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not load image from {image_path}")
        sys.exit(1)

    # Now lines will be just the two longest red lines
    lines = detector_v4(image)
    clusters = cluster_lines(lines)
    for cluster in clusters.values():
        print(cluster)

    # Draw lines
    output = image.copy()
    for idx, (x1, y1, x2, y2, m) in enumerate(lines):
        slope = (y2 - y1) / (x2 - x1)
        # If slope is positive, draw red (BGR -> (0, 0, 255)), if negative, blue (255, 0, 0)
        if slope > 0:
            color = (0, 0, 255)  # red
        else:
            color = (255, 0, 0)  # blue
        cv2.line(output, (x1, y1), (x2, y2), color, 2)

    print(f"Detected {len(lines)} line(s).")

    # 1. Compute intersections for each pair of lines
    intersections = []
    n_lines = len(lines)
    for i in range(n_lines):
        for j in range(i + 1, n_lines):
            point = line_segment_intersection(lines[i], lines[j])
            if point is not None:
                intersections.append(point)
                print(f"Intersection of line {i} with line {j} at ("
                      f"{round(point[0], 2)}, {round(point[1], 2)})")

    # 2. Check for duplicate intersections (rounded)
    intersection_counts = {}
    for pt in intersections:
        rounded = (round(pt[0], 2), round(pt[1], 2))
        intersection_counts[rounded] = intersection_counts.get(rounded, 0) + 1

    duplicates = {k: v for k, v in intersection_counts.items() if v > 1}
    if duplicates:
        print("\nDuplicate intersections found:")
        for k, v in duplicates.items():
            print(f"  Intersection {k} appears {v} times.")
    else:
        print("\nNo duplicate intersections found.")

    print("Press any key to close the result window.")

    # Show output
    cv2.imshow("Detector v4 - Hough Lines", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


'''