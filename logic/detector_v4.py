#!/usr/bin/env python3

import sys
import cv2
import numpy as np

from logic.line_clustering import cluster_lines


def detector_v4(input_image):
    """
    Similar to detector_v2, this function accepts an input image and returns
    two lines (each line is [x1, y1, x2, y2, slope]) that:
      1) Are among the top 25% longest positive-slope lines detected.
      2) Are the farthest apart by any endpoint distance 
         (so we don't just detect two portions of the same line).
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
            if slope > 0:
                # Compute line length
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                red_lines.append([x1, y1, x2, y2, slope, length])

    # Sort red_lines by length (desc), then keep only the top 25%
    red_lines.sort(key=lambda ln: ln[5], reverse=True)
    if len(red_lines) == 0:
        return []  # no lines at all

    num_top_25 = max(1, int(len(red_lines) * 0.5))
    top_25pct_lines = red_lines[:num_top_25]

    # If we only have one line in top 25%, just return that single line (minus the length field)
    if len(top_25pct_lines) < 2:
        return [[ln[0], ln[1], ln[2], ln[3], ln[4]] for ln in top_25pct_lines]

    # --------------------- REPLACE midpoint LOGIC WITH ENDPOINTS LOGIC --------------------- #
    def endpoints_max_dist(lineA, lineB):
        """
        Returns the maximum distance among any combination of endpoints
        from lineA and lineB. Each line is [x1, y1, x2, y2, slope, length].
        """
        x1A, y1A, x2A, y2A, slopeA, lenA = lineA
        x1B, y1B, x2B, y2B, slopeB, lenB = lineB

        endpointsA = [(x1A, y1A), (x2A, y2A)]
        endpointsB = [(x1B, y1B), (x2B, y2B)]

        max_d = 0
        for (xa, ya) in endpointsA:
            for (xb, yb) in endpointsB:
                dist = np.hypot(xb - xa, yb - ya)
                if dist > max_d:
                    max_d = dist
        return max_d

    max_dist = -1
    best_pair = None
    for i in range(len(top_25pct_lines)):
        for j in range(i + 1, len(top_25pct_lines)):
            dist = endpoints_max_dist(top_25pct_lines[i], top_25pct_lines[j])
            if dist > max_dist:
                max_dist = dist
                best_pair = (top_25pct_lines[i], top_25pct_lines[j])

    # best_pair holds two lines that are farthest apart by endpoint distance
    lineA, lineB = best_pair
    final_lines = []
    for ln in [lineA, lineB]:
        x1, y1, x2, y2, slope, length = ln
        final_lines.append([x1, y1, x2, y2, slope])

    return final_lines


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

    # lines now contains exactly two lines (or fewer if top-25% < 2 lines)
    lines = detector_v4(image)
    print(f"Detected {len(lines)} final line(s).")

    # Draw lines
    output = image.copy()
    for (x1, y1, x2, y2, slope) in lines:
        color = (0, 0, 255)  # red
        cv2.line(output, (x1, y1), (x2, y2), color, 1)

    cv2.imshow("detector_v4 - final lines", output)
    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
'''

