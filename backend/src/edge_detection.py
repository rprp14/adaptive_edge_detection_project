import cv2
import os
from src.feature_extraction import extract_features
from src.ml_predictor import predict_threshold
from src.pso_optimizer import optimize_threshold

# Ensure output directories exist
os.makedirs('outputs/basic', exist_ok=True)
os.makedirs('outputs/adaptive', exist_ok=True)
os.makedirs('outputs/pso', exist_ok=True)

def basic_edge_detection(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image {image_path}")
        
        edges = cv2.Canny(img, 100, 200)
        output_path = f'outputs/basic/{os.path.basename(image_path)}'
        cv2.imwrite(output_path, edges)
        return output_path
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def adaptive_edge_detection(image_path):
    try:
        features = extract_features(image_path)
        threshold = predict_threshold(features)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image {image_path}")
        
        lower_thresh = max(0, threshold - 30)
        upper_thresh = min(255, threshold + 30)
        edges = cv2.Canny(img, lower_thresh, upper_thresh)
        
        output_path = f'outputs/adaptive/{os.path.basename(image_path)}'
        cv2.imwrite(output_path, edges)
        return output_path
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def pso_edge_detection(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image {image_path}")

        def cost(thresh):
            edges = cv2.Canny(img, thresh[0], thresh[0] + 50)
            return -edges.sum()

        best_thresh = optimize_threshold(cost, [50], [150], max_iter=100, max_particles=30)
        lower_thresh = max(0, best_thresh - 20)
        upper_thresh = min(255, best_thresh + 20)
        edges = cv2.Canny(img, lower_thresh, upper_thresh)
        
        output_path = f'outputs/pso/{os.path.basename(image_path)}'
        cv2.imwrite(output_path, edges)
        return output_path
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None
