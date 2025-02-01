#!/usr/bin/env python3

import sys
import cv2
import numpy as np

from sklearn.cluster import KMeans  # pip install scikit-learn

def main():
    if len(sys.argv) < 2:
        print("Usage: python clustered_lines_by_slope.py <image_path>")
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
    lines = cv2.HoughLinesP(
        canny,
        1,                # rho resolution
        np.pi / 180,      # theta resolution
        50,               # threshold
        minLineLength=100,
        maxLineGap=20
    )

    # We'll cluster lines purely by slope (angle).
    # Then draw them in different colors based on cluster.

    # Prepare a copy for clustering visualization
    cluster_output = frame.copy()

    if lines is None or len(lines) == 0:
        print("No lines found via HoughLinesP.")
        # Show the masked image anyway
        cv2.imshow("Masked to Green Field", frame_masked)
        cv2.imshow("Clustered Lines (None found)", cluster_output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Build a feature vector: [ slope ]
    # slope = (y2 - y1)/(x2 - x1), or a large number for vertical lines
    slopes = []
    lines_data = []  # store (x1, y1, x2, y2) for each line

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            slope = 9999.0  # represent vertical lines with a large slope
        else:
            slope = dy / float(dx)

        slopes.append([slope])
        lines_data.append((x1, y1, x2, y2))
    print(slopes)
    slopes = np.array(slopes, dtype=np.float32)

    #
    # 2) K-Means Clustering by slope
    #
    # For a field with mostly horizontal & vertical lines, you might try n_clusters=2
    # If you have diagonal lines as well, try n_clusters=3 or more.
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(slopes)

    # We'll define a set of colors for up to 5 clusters; adjust as needed
    color_table = [
        (0, 0, 255),    # red
        (0, 255, 0),    # green
        (255, 0, 0),    # blue
        (0, 255, 255),  # yellow
        (255, 0, 255),  # magenta
    ]

    # Draw lines in different colors based on cluster label
    for i, (x1, y1, x2, y2) in enumerate(lines_data):
        cluster_id = labels[i]
        color = color_table[cluster_id % len(color_table)]
        cv2.line(cluster_output, (x1, y1), (x2, y2), color, 2)

    # Show windows
    cv2.imshow("Masked to Green Field", frame_masked)
    cv2.imshow("Clustered Lines by Slope", cluster_output)

    print(f"Found {len(lines)} lines, clustered into {n_clusters} groups by slope.")
    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
