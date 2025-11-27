# FILE: backend/train_unet_model.py
# PURPOSE: Script to train the standard UNet model for edge detection.

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# --- 0. Custom Loss Function Fix (Required for consistent training) ---
# We use the same loss function as the ViT model for consistency.
def CustomBinaryCrossentropy(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true_flat = tf.reshape(y_true, shape=[-1])
    y_pred_flat = tf.reshape(y_pred, shape=[-1])
    return tf.keras.losses.binary_crossentropy(y_true_flat, y_pred_flat)

# --- 1. Model Definition (Standard UNet) ---
def conv_block(input_tensor, num_filters):
    x = layers.Conv2D(num_filters, 3, padding="same", activation="relu")(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(num_filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    return x

def build_unet_model(input_shape=(256, 256, 1)):
    inputs = layers.Input(shape=input_shape, name="input_layer")

    # --- ENCODER ---
    c1 = conv_block(inputs, 16)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = conv_block(p1, 32)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = conv_block(p2, 64)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    c4 = conv_block(p3, 128)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    
    # --- BOTTLENECK ---
    c5 = conv_block(p4, 256)
    
    # --- DECODER ---
    u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = conv_block(u6, 128)
    
    u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = conv_block(u7, 64)
    
    u8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = conv_block(u8, 32)

    u9 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = conv_block(u9, 16)
    
    # Final 1x1 convolution for single channel output (edge mask)
    outputs = layers.Conv2D(1, 1, activation="sigmoid", padding="same", name="final_mask_conv")(c9)
    
    model = Model(inputs=inputs, outputs=outputs, name="UNet_Edge_Detector")
    model.compile(optimizer='adam', loss=CustomBinaryCrossentropy, metrics=['accuracy'])
    return model

# --- 2. Data Loading Function (Copied from train_edge_model.py for consistency) ---
def load_data(image_path, mask_path, img_width, img_height):
    print("Loading data...")
    images = []
    masks = []
    
    image_files = sorted([f for f in os.listdir(image_path) if not f.startswith('.')])
    mask_files = sorted([f for f in os.listdir(mask_path) if not f.startswith('.')])
    
    common_files = sorted(list(set(image_files) & set(mask_files)))
    
    if len(common_files) == 0:
         print("Error: No common images/masks found. Check paths and content.")
         return None, None
    
    for filename in common_files:
        try:
            img = cv2.imread(os.path.join(image_path, filename), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(os.path.join(mask_path, filename), cv2.IMREAD_GRAYSCALE)
            
            if img is None or mask is None:
                continue

            img = cv2.resize(img, (img_height, img_width)) / 255.0
            mask = cv2.resize(mask, (img_height, img_width)) / 255.0
            
            images.append(np.expand_dims(img, axis=-1).astype(np.float32))
            masks.append(np.expand_dims(mask, axis=-1).astype(np.float32)) 
            
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            
    if not images:
        print("Error: All images failed processing.")
        return None, None
            
    X_data = np.stack(images, axis=0)
    y_data = np.stack(masks, axis=0) 
    
    print(f"Data loaded successfully! Found {len(X_data)} images/masks.")
    return X_data, y_data


# --- 3. Main Execution ---
if __name__ == "__main__":
    
    # --- Configuration ---
    IMAGE_PATH = 'backend/dataset'
    MASK_PATH = 'backend/canny_masks'
    MODEL_SAVE_PATH = 'backend/models/edge_detection_model.h5' # Save path used by app.py
    
    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    IMG_CHANNELS = 1 
    BATCH_SIZE = 4 
    EPOCHS = 20 # You can adjust this

    X, y = load_data(IMAGE_PATH, MASK_PATH, IMG_WIDTH, IMG_HEIGHT)

    if X is not None and y is not None:
        
        # 1. Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

        input_shape = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
        
        # --- BUILD THE UNET MODEL ---
        model = build_unet_model(input_shape=input_shape)
        model.summary()

        print("Starting UNet model training...")
        
        # 2. Fit the model
        model.fit(
            X_train,        
            y_train,
            validation_data=(X_val, y_val), 
            batch_size=BATCH_SIZE, 
            epochs=EPOCHS 
        )

        model.save(MODEL_SAVE_PATH)
        print(f"\n--- UNet Model training complete! ---")
        print(f"New UNet model saved to: {MODEL_SAVE_PATH}")
    else:
        print("Could not load data. Please check paths and file integrity.")