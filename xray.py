import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Define model path
MODEL_PATH = "my_model.h5"  # Ensure the model file is in the correct directory

# Load TensorFlow model
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model Loaded Successfully.")
    print(f"🔹 Model Expected Input Shape: {model.input_shape}")  # Debugging
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None  # Ensure the app doesn't crash if the model fails to load

# Define allowed file types
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Function to check file extension
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Image Preprocessing Function
def preprocess_image(image):
    """
    Converts an input image to the model's expected format (grayscale or RGB), resizes it, 
    normalizes pixel values, and reshapes it accordingly.
    """
    try:
        expected_shape = model.input_shape  # Get model's expected shape
        num_channels = expected_shape[-1]  # Get expected number of channels

        # Convert to correct color mode
        if num_channels == 1:
            image = image.convert("L")  # Convert to grayscale (1 channel)
        else:
            image = image.convert("RGB")  # Convert to RGB (3 channels)

        image = image.resize((224, 224))  # Resize to match model input
        image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize pixel values

        # Reshape to match model input (Batch Size, Height, Width, Channels)
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        if num_channels == 1:
            image_array = np.expand_dims(image_array, axis=-1)  # Ensure shape (1, 224, 224, 1)

        print(f"✅ Preprocessed Image Shape: {image_array.shape}")  # Debugging
        return image_array
    except Exception as e:
        print(f"❌ Error in Image Preprocessing: {e}")
        return None

@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles image upload, preprocesses the image, runs it through the model, 
    and returns the predicted class and confidence score.
    """
    if model is None:
        return jsonify({"error": "Model not loaded properly."}), 500

    try:
        # Ensure an image is uploaded
        if "xray_image" not in request.files:
            return jsonify({"error": "No image file uploaded"}), 400
        
        file = request.files["xray_image"]

        # Validate file
        if file.filename == "" or not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Allowed: PNG, JPG, JPEG"}), 400

        # Read and preprocess the image
        image = Image.open(file)
        processed_image = preprocess_image(image)

        if processed_image is None:
            return jsonify({"error": "Error processing image"}), 500

        # Make prediction
        prediction = model.predict(processed_image)
        predicted_label = int(np.argmax(prediction))  # Assuming classification model
        confidence_score = float(np.max(prediction))

        # Response
        return jsonify({
            "diseases": f"Disease {predicted_label}",
            "confidence": f"{confidence_score:.2f}"
        })

    except Exception as e:
        print(f"❌ Server Error: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, threaded=True)  # Enables handling multiple requests
