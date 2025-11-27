# FILE: backend/train_edge_model.py
# PURPOSE: Final code for training the Hybrid ViT-UNet model (Fixed).

import tensorflow as tf
from tensorflow.keras.models import Model
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from vit_model import build_vit_segmentation_model 

# --- 1. Set up Paths ---
IMAGE_PATH = 'backend/dataset'
MASK_PATH = 'backend/canny_masks'
MODEL_SAVE_PATH = 'backend/models/vit_unet_model.h5' 

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 1 
BATCH_SIZE = 4 
EPOCHS = 20

# --- 2. Data Loading and Preprocessing Function (Stable Structure) ---
def load_data(image_path, mask_path):
    print("Loading data...")
    images = []
    masks = []
    
    # Note: The "Premature end of JPEG file" warnings are likely due to corrupted 
    # or incomplete JPEG files in your dataset, but the code is designed to skip them.
    # We will ignore these warnings and proceed with the rest of the data.
    
    image_files = sorted([f for f in os.listdir(image_path) if not f.startswith('.')])
    mask_files = sorted([f for f in os.listdir(mask_path) if not f.startswith('.')])
    
    # Filter files that exist in both directories (assuming file names match)
    common_files = sorted(list(set(image_files) & set(mask_files)))
    
    if len(common_files) == 0:
         print("Error: No common images/masks found. Check paths and content.")
         return None, None
    
    for filename in common_files:
        try:
            img = cv2.imread(os.path.join(image_path, filename), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(os.path.join(mask_path, filename), cv2.IMREAD_GRAYSCALE)
            
            # Ensure images loaded correctly
            if img is None or mask is None:
                # print(f"Warning: Skipping {filename} due to read error.")
                continue

            # Resize and Normalize
            img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH)) / 255.0
            mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH)) / 255.0
            
            # Input (X) and Target (Y) must be (H, W, 1)
            images.append(np.expand_dims(img, axis=-1).astype(np.float32))
            masks.append(np.expand_dims(mask, axis=-1).astype(np.float32)) 
            
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            
    if not images:
        print("Error: All images failed processing.")
        return None, None
            
    # Final arrays are shaped (N, H, W, 1)
    X_data = np.stack(images, axis=0)
    y_data = np.stack(masks, axis=0) 
    
    print(f"Data loaded successfully! Found {len(X_data)} images/masks.")
    return X_data, y_data


# --- 3. Main Execution: Load, Build, Train ---
if __name__ == "__main__":
    
    X, y = load_data(IMAGE_PATH, MASK_PATH)

    if X is not None and y is not None:
        
        # 1. Split data (NumPy arrays)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

        input_shape = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
        
        # --- BUILD THE HYBRID ViT MODEL ---
        # This function call now works correctly with the fixed vit_model.py
        model = build_vit_segmentation_model(input_shape=input_shape)
        model.summary()

        print("Starting Hybrid ViT-UNet model training...")
        
        # 2. Fit the model using NumPy arrays (The most stable method)
        model.fit(
            X_train,        
            y_train,
            validation_data=(X_val, y_val), 
            batch_size=BATCH_SIZE, 
            epochs=EPOCHS 
        )

        model.save(MODEL_SAVE_PATH)
        print(f"\n--- Hybrid ViT Model training complete! ---")
        print(f"New ViT model saved to: {MODEL_SAVE_PATH}")
    else:
        print("Could not load data. Please check paths and file integrity.")