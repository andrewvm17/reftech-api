import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

def get_prominent_colors(image_path, K=7, show_plot=False, apply_blur=True):
    """
    Detect prominent colors in an image using k-means clustering.
    
    Parameters:
        image_path (str): Path to the image file.
        K (int): Number of clusters. Increasing this can help detect more subtle differences.
        show_plot (bool): If True, display a plot of the color distribution.
        apply_blur (bool): If True, apply a Gaussian blur to reduce noise.
        
    Returns:
        centers (np.array): Array of cluster centers in HSV.
        counts (np.array): Pixel counts for each cluster.
        dominant_color (np.array): The most prominent cluster center (HSV).
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    
    # Optionally apply a slight Gaussian blur to reduce noise
    if apply_blur:
        image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Convert from BGR (OpenCV default) to HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Reshape image into a list of pixels (each pixel is a 3-dimensional point in HSV)
    pixels = image_hsv.reshape((-1, 3))
    pixels = np.float32(pixels)

    # Define criteria for k-means:
    # - Use a much lower epsilon (0.01 instead of 0.2) to make clustering more sensitive to small differences.
    # - Increase maximum iterations from 100 to 200.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.01)
    
    # Run k-means clustering
    ret, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Count how many pixels were assigned to each cluster
    labels = labels.flatten()
    counts = np.bincount(labels)

    # Identify the cluster with the most pixels (the most prominent color)
    dominant_idx = np.argmax(counts)
    dominant_color = centers[dominant_idx]
    
    # Convert centers to integers for easier interpretation (if desired)
    centers_int = np.uint8(centers)
    
    # Optionally plot the color distribution (as bars) to visualize the clusters
    if show_plot:
        plt.figure(figsize=(8, 4))
        bar = np.zeros((50, 300, 3), dtype=np.uint8)
        start_x = 0
        for i in range(K):
            # Calculate the width of each color bar proportionally to the count of pixels
            end_x = start_x + (counts[i] / np.sum(counts)) * 300
            cv2.rectangle(bar, (int(start_x), 0), (int(end_x), 50), 
                          color=tuple(int(x) for x in centers_int[i]), thickness=-1)
            start_x = end_x
        # Convert bar image from HSV to RGB for correct color display
        bar_rgb = cv2.cvtColor(bar, cv2.COLOR_HSV2RGB)
        plt.imshow(bar_rgb)
        plt.title("Color Distribution (HSV clusters)")
        plt.axis("off")
        plt.show()

    return centers, counts, dominant_color

# Example usage:
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python color_detector_v0.py <image_path> [K=7]")
        sys.exit(1)

    image_path = sys.argv[1]

    # Use K=7 (or a higher number) to better separate subtle color differences
    K = 7
    if len(sys.argv) == 3:
        try:
            K = int(sys.argv[2])
        except ValueError:
            print("K must be an integer. Using default K=7.")

    # Run the color detection with more sensitive parameters
    centers, counts, field_color = get_prominent_colors(image_path, K=K, show_plot=True, apply_blur=True)

    print("\nCluster centers (in HSV):")
    for idx, center in enumerate(centers):
        print(f"Cluster {idx}: {center}, Count: {counts[idx]}")

    print("\nAssumed field (most prominent) color (HSV):")
    print(field_color)
