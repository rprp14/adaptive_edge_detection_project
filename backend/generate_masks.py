# FILE: backend/generate_masks.py
# PURPOSE: To create the 'ground truth' Canny masks from your dataset.
# RUN THIS SCRIPT *FIRST* (after adding images to backend/dataset/).

import os
import cv2
import glob

# --- 1. Set Paths (Corrected for your project) ---
IMAGE_SOURCE_PATH = 'backend/dataset' 
MASK_OUTPUT_PATH = 'backend/canny_masks'

# --- 2. Canny Edge Parameters (You can tune these) ---
LOW_THRESHOLD = 50
HIGH_THRESHOLD = 150

def create_ground_truth():
    print(f"Source Image Path: {IMAGE_SOURCE_PATH}")
    print(f"Mask Output Path: {MASK_OUTPUT_PATH}")

    # Create output directory if it doesn't exist
    os.makedirs(MASK_OUTPUT_PATH, exist_ok=True)

    # Find all .png or .jpg images
    image_files = glob.glob(os.path.join(IMAGE_SOURCE_PATH, '*.png'))
    image_files.extend(glob.glob(os.path.join(IMAGE_SOURCE_PATH, '*.jpg')))
    
    if not image_files:
        print(f"Error: No images found in '{IMAGE_SOURCE_PATH}'.")
        print("Please add your training X-ray images to that folder.")
        return

    print(f"Found {len(image_files)} images. Starting processing...")

    for img_path in image_files:
        try:
            filename = os.path.basename(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"Warning: Could not read {filename}. Skipping.")
                continue

            edges = cv2.Canny(img, LOW_THRESHOLD, HIGH_THRESHOLD)
            output_path = os.path.join(MASK_OUTPUT_PATH, filename)
            cv2.imwrite(output_path, edges)
            
            print(f"Successfully created mask for {filename}.")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print("--- Ground truth generation complete! ---")

if __name__ == "__main__":
    create_ground_truth()