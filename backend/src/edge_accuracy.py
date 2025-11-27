import cv2
import numpy as np

def compute_edge_accuracy(edge_image):
    """
    Computes clarity score for detected edges (0–100%).
    Higher score means clearer and stronger edges.
    """

    # Convert to grayscale if needed
    if len(edge_image.shape) == 3:
        edge_image = cv2.cvtColor(edge_image, cv2.COLOR_BGR2GRAY)

    # Threshold to get strong edges
    _, binary = cv2.threshold(edge_image, 50, 255, cv2.THRESH_BINARY)

    edge_pixels = np.sum(binary > 0)
    total_pixels = binary.size

    if total_pixels == 0:
        return 0.0

    ratio = edge_pixels / total_pixels
    score = round(min(max(ratio * 100, 0), 100), 2)

    return score
