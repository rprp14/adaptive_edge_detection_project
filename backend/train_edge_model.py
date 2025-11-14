import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

def build_model():
    model = models.Sequential([
        layers.InputLayer(input_shape=(256, 256, 3)),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.UpSampling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.UpSampling2D((2,2)),
        layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def dummy_train_model():
    model = build_model()

    # Dummy data (random images and random edges)
    X_train = np.random.rand(10, 256, 256, 3)
    Y_train = np.random.randint(0, 2, size=(10, 256, 256, 1))

    model.fit(X_train, Y_train, batch_size=2, epochs=2)
    model.save('edge_detection_model.h5')
    print("✅ Model trained and saved as edge_detection_model.h5")

if __name__ == '__main__':
    dummy_train_model()
