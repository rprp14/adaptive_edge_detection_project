import numpy as np
import pickle
import os

# Correct model path
model_path = os.path.join(os.getcwd(), 'models', 'threshold_predictor.pkl')

print(f"Model path: {model_path}")
with open(model_path, 'rb') as f:
    model = pickle.load(f)


print("Model loaded successfully.")
def predict_threshold(features):
    """
    Predict the threshold using the trained model.
    
    Args:
        features (list or numpy array): Input features for prediction (must match training features shape).

    Returns:
        Predicted threshold value (e.g., 0 or 1 for binary, or continuous depending on model)
    """
    features = np.array(features).reshape(1, -1)  # Make sure features are 2D
    prediction = model.predict(features)
    return prediction[0]
