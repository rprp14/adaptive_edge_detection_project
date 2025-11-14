import cv2
import numpy as np

def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mean = np.mean(img)
    std = np.std(img)
    var = np.var(img)
    return np.array([mean, std, var])
