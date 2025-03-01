# detection.py
import tensorflow as tf
import numpy as np
import os
from PIL import Image

def load_detection_model(model_path='C:/Users/Yadav Ji/Desktop/BrainAI_framework/BrainAI_Framework/models/Lenet_model.h5'):
    """Loads a detection model from an .h5 file."""
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        return model
    else:
        print(f"Error: Model not found at {model_path}")
        return None

def preprocess_detection_image(image):
    """Preprocesses an image for detection."""
    img = image.resize((224, 224))  # Adjust size as needed
    img_array = np.array(img) / 255.0  # Normalize
    return img_array

def detect_tumor(image, model):
    """Performs tumor detection on an image."""
    processed_image = preprocess_detection_image(image)
    prediction = model.predict(np.expand_dims(processed_image, axis=0))
    return prediction