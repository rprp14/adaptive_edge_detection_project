'''import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier  # Example model

# Example: Replace this with your actual training data
X_train = np.random.rand(100, 5)  # Features: 100 samples, 5 features each
y_train = np.random.randint(0, 2, size=100)  # Labels: binary classification

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model to backend/models/threshold_predictor.pkl
model_path = 'backend/models/threshold_predictor.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"Model saved at: {model_path}")
'''

#===============================================================================================================================
# FILE: backend/train_model.py
# PURPOSE: Updated version of your existing script
# TRAIN RandomForest using 3 REAL features (mean, std, entropy)

import pickle
import numpy as np
import cv2
import glob
from sklearn.ensemble import RandomForestClassifier
from skimage.measure import shannon_entropy


# ---------------------------------------------
# FUNCTION: Extract 3 real features from an image
# ---------------------------------------------
def extract_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Warning: Could not read {img_path}. Skipping.")
        return None

    img = cv2.resize(img, (256, 256))

    mean = np.mean(img)
    std = np.std(img)
    entropy = shannon_entropy(img)

    return [mean, std, entropy]


# ---------------------------------------------
# LOAD DATASET (REPLACES RANDOM DATA IN OLD CODE)
# ---------------------------------------------
image_paths = glob.glob("backend/dataset/*.*")

if len(image_paths) == 0:
    print("❌ ERROR: No training images found in backend/dataset/")
    exit()

X_train = []
y_train = []

print("Extracting features from training images...")

for path in image_paths:
    feats = extract_features(path)

    if feats is None:
        continue

    X_train.append(feats)

    # ---------------------------------------------
    # NOTE: Replace 1 with REAL labels if available
    # Currently using dummy label (same as your old code)
    # ---------------------------------------------
    y_train.append(1)

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"✔ Total samples used for training: {len(X_train)}")
print("✔ Training RandomForest model...")


# ---------------------------------------------
# TRAIN THE MODEL (same as your old code)
# ---------------------------------------------
model = RandomForestClassifier()
model.fit(X_train, y_train)

# ---------------------------------------------
# SAVE MODEL (same location as your old code)
# ---------------------------------------------
model_path = 'backend/models/threshold_predictor.pkl'

with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"🎉 Model saved at: {model_path}")
