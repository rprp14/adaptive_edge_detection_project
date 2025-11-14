import pickle
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
