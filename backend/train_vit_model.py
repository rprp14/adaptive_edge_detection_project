# FILE: backend/train_vit_model.py
# PURPOSE: Train Hybrid ViT-UNet for edge detection and save vit_unet_weights.h5 + vit_unet_model.h5

import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Import ViT-UNet Builder
from vit_model import build_vit_segmentation_model

# ------------------ Paths ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MASKS_DIR = os.path.join(BASE_DIR, "canny_masks")

MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

VIT_MODEL_SAVE = os.path.join(MODEL_OUTPUT_DIR, "vit_unet_model.h5")
VIT_WEIGHTS_SAVE = os.path.join(MODEL_OUTPUT_DIR, "vit_unet_weights.h5")

# ------------------ Training Config ------------------
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 1
BATCH_SIZE = 4
EPOCHS = 20

# ------------------ Load Image + Mask Data ------------------
def load_data():
    images = []
    masks = []

    image_files = sorted([f for f in os.listdir(DATASET_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    mask_files = sorted([f for f in os.listdir(MASKS_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    common_files = list(set(image_files) & set(mask_files))
    common_files.sort()

    if len(common_files) == 0:
        print("ERROR: No matching image–mask pairs found.")
        return None, None

    print(f"Found {len(common_files)} training image-mask pairs.")

    for fname in common_files:
        try:
            img_path = os.path.join(DATASET_DIR, fname)
            mask_path = os.path.join(MASKS_DIR, fname)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if img is None or mask is None:
                print(f"Skipping corrupted: {fname}")
                continue

            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
            mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT)) / 255.0

            images.append(np.expand_dims(img, axis=-1))
            masks.append(np.expand_dims(mask, axis=-1))

        except Exception as e:
            print(f"Error loading {fname}: {e}")

    if len(images) == 0:
        print("ERROR: No valid training images.")
        return None, None

    return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)

# ------------------ Custom Loss ------------------
def CustomBinaryCrossentropy(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.keras.losses.binary_crossentropy(tf.reshape(y_true, [-1]), tf.reshape(y_pred, [-1]))

# ------------------ Main Training Pipeline ------------------
if __name__ == "__main__":

    print("\n=== LOADING DATASET FOR ViT-UNET TRAINING ===")
    X, y = load_data()

    if X is None or y is None:
        exit()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")

    # ------------------ Build ViT-UNet ------------------
    print("\n=== BUILDING ViT-UNET MODEL ===")

    model = build_vit_segmentation_model(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
        patch_size=16,
        num_layers=4,
        num_heads=4,
        embed_dim=64,
        feed_forward_dim=128,
        decoder_filters=(64, 32, 16, 8)
    )

    model.compile(
        optimizer="adam",
        loss=CustomBinaryCrossentropy,
        metrics=["accuracy"]
    )

    model.summary()

    # ------------------ TRAINING ------------------
    print("\n=== TRAINING STARTED ===\n")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )

    # ------------------ SAVE FULL MODEL ------------------
    print("\n=== SAVING FULL MODEL AND WEIGHTS ===")

    model.save(VIT_MODEL_SAVE)
    print(f"Saved full ViT-UNet model to: {VIT_MODEL_SAVE}")

    model.save_weights(VIT_WEIGHTS_SAVE)
    print(f"Saved ViT-UNet weights to: {VIT_WEIGHTS_SAVE}")

    print("\n=== TRAINING COMPLETED SUCCESSFULLY ===")
