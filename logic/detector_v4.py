#!/usr/bin/env python3

import sys
import cv2
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("Usage: python line_comparison.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error loading image: {image_path}")
        sys.exit(1)

    # Convert BGR -> HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a green mask (tweak these values for your field's lighting)
    lower_green = (36, 25, 25)
    upper_green = (86, 255, 255)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Keep only the green field in 'frame_masked'
    frame_masked = cv2.bitwise_and(frame, frame, mask=mask_green)

    # Convert to grayscale
    gray = cv2.cvtColor(frame_masked, cv2.COLOR_BGR2GRAY)

    # Simple binary threshold (optional)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Canny edge detection
    canny = cv2.Canny(thresh, 50, 150, apertureSize=3)

    #
    # 1) Hough Line Detection (Probabilistic)
    #
    # lines: list of [x1, y1, x2, y2]
    lines = cv2.HoughLinesP(
        canny,
        1,                # rho resolution
        np.pi / 180,      # theta resolution
        50,               # threshold
        minLineLength=150,
        maxLineGap=20
    )

    # Create an output for Hough lines
    hough_output = frame.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate the slope
            if x2 - x1 != 0:  # Avoid division by zero
                slope = (y2 - y1) / (x2 - x1)
                
                # Ignore purely vertical (undefined slope) and horizontal (slope = 0) lines
                if slope != 0:
                    print(slope)

                    cv2.line(hough_output, (x1, y1), (x2, y2), (0, 0, 255), 2)
            else:
                # Ignore purely vertical lines
                continue
    else:
        print("No lines found via HoughLinesP.")

    #
    # 2) Line Segment Detector (LSD)
    #
    # Option A: If you have the ximgproc module:
    # from cv2.ximgproc import createFastLineDetector
    # lsd = createFastLineDetector()

    # Option B: If your OpenCV includes the built-in LSD:
    lsd = cv2.createLineSegmentDetector(0)  # 0=refine_none, see docs for more modes
    lines_lsd = lsd.detect(canny)[0]        # returns a list of lines

    lsd_output = frame.copy()
    if lines_lsd is not None:
        # drawSegments expects lines in Nx1x4 shape -> (x1,y1,x2,y2)
        lsd.drawSegments(lsd_output, lines_lsd)
    else:
        print("No line segments found via LSD.")

    # Show everything
    #cv2.imshow("Original", frame)
    cv2.imshow("Masked to Green Field", frame_masked)
    #cv2.imshow("Canny Edges", canny)
    cv2.imshow("Hough Lines", hough_output)
    #cv2.imshow("LSD Lines", lsd_output)

    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
