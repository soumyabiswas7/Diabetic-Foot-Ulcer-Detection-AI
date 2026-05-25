# foot_ulcer_inference.py
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

model = load_model('foot_ulcer_model_efficientnet.h5')

# (same as used during training, e.g., 224x224)
IMG_SIZE = (224, 224)  # Width, Height

# Function to preprocess any input image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)  # Resizes automatically
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to predict foot ulcer
def predict(img_path):
    preprocessed_img = preprocess_image(img_path)
    prediction = model.predict(preprocessed_img)
    if prediction[0][0] >= 0.5:
        result = "Ulcer"
        confidence = prediction[0][0]
    else:
        result = "No Ulcer"
        confidence = 1 - prediction[0][0]
    return result, confidence

if __name__ == "__main__":
    test_image_path = input("Enter the path of the foot image: ")
    if not os.path.exists(test_image_path):
        print(f"Image not found: {test_image_path}")
    else:
        label, conf = predict(test_image_path)
        print(f"Prediction: {label} | Confidence: {conf:.2f}")

