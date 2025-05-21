import torch
import torch.nn as nn # Might be needed if your model uses specific PyTorch layers for loading
import torchvision.transforms as transforms
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import os
import json
import pickle # If you load labels from a pickle file

from flask import Flask, request, jsonify

# --- Configuration ---
# Adjust these paths if your 'utils' directory is somewhere else
ONNX_MODEL_PATH = "C:\\Users\\navsh\\OneDrive\\Documents\\GitHub\\animalrec\\models\\actualfinal.onnx"
LABELS_FILE_PATH = "C:\\Users\\navsh\\OneDrive\\Documents\\GitHub\\animalrec\\models\\labels.json" # Path to your saved labels JSON
IMG_SIZE = 224 # This MUST match the input size your ONNX model expects

# --- Global variables to hold loaded model and labels ---
ort_session = None
LABELS_INVERSE = {} # To map predicted index back to class name

# --- Function to load model and labels at service startup ---
def load_resources():
    global ort_session
    global LABELS_INVERSE

    # 1. Load ONNX Model
    try:
        if not os.path.exists(ONNX_MODEL_PATH):
            raise FileNotFoundError(f"ONNX model not found at: {ONNX_MODEL_PATH}. "
                                    "Please ensure you have trained and exported your model.")
        ort_session = ort.InferenceSession(ONNX_MODEL_PATH)
        print(f"ONNX model loaded successfully from {ONNX_MODEL_PATH}")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load ONNX model: {e}")
        ort_session = None # Ensure it's None if loading fails

    # 2. Load LABELS
    try:
        if os.path.exists(LABELS_FILE_PATH):
            with open(LABELS_FILE_PATH, "r") as f:
                labels = json.load(f)
            LABELS_INVERSE = {v: k for k, v in labels.items()}
            print(f"LABELS loaded successfully from {LABELS_FILE_PATH}")
        else:
            # Fallback if labels.json not found, perhaps from a pickle or hardcode
            print(f"WARNING: {LABELS_FILE_PATH} not found. Attempting to load from training_data.pkl or using dummy labels.")
            if os.path.exists("utils/training_data.pkl"):
                with open("utils/training_data.pkl", "rb") as f:
                    training_data = pickle.load(f)
                # This assumes your pickle contains [image_path, label_index] pairs
                unique_indices = sorted(list(set([item[1] for item in training_data])))
                # This is a dangerous fallback if you don't know the actual class names
                # It's HIGHLY recommended to explicitly save labels.json during training
                # Example: {0: "class_0", 1: "class_1"}
                LABELS_INVERSE = {idx: f"unknown_class_{idx}" for idx in unique_indices}
                print(f"LABELS derived (partially) from training_data.pkl: {LABELS_INVERSE}")
            else:
                print("WARNING: No labels file or training_data.pkl found. Using hardcoded dummy labels.")
                LABELS_INVERSE = {0: "dummy_class_0", 1: "dummy_class_1", 2: "dummy_class_2"} # Add more if your model outputs more classes
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load LABELS: {e}. Using dummy labels.")
        LABELS_INVERSE = {0: "fallback_label"} # Last resort fallback

# --- Image Preprocessing Function (adapted from your script) ---
def preprocess_image_from_bytes(image_bytes):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0) # Add batch dimension
    return img_tensor.numpy() # Convert to NumPy array for ONNX Runtime

# --- Flask App Setup ---
app = Flask(__name__)

# Basic Health Check (good practice for microservices)
@app.route('/health', methods=['GET'])
def health_check():
    status = 'healthy' if ort_session is not None and LABELS_INVERSE else 'unhealthy'
    return jsonify({'status': status, 'model_loaded': ort_session is not None, 'labels_loaded': bool(LABELS_INVERSE)}), 200

# Prediction Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if not ort_session:
        return jsonify({'error': 'Model not loaded, prediction service not ready.'}), 503 # Service Unavailable

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided. Please send a file with key "image".'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image_bytes = file.read()
        image_np_array = preprocess_image_from_bytes(image_bytes)

        # Get input name from ONNX session (robust way)
        input_name = ort_session.get_inputs()[0].name
        inputs = {input_name: image_np_array}

        # Run inference
        outputs = ort_session.run(None, inputs)

        # Post-processing (from your original script)
        predictions = outputs[0][0] # Assuming output[0] is your prediction tensor
        predicted_index = predictions.argmax()

        # Get label from loaded LABELS_INVERSE
        predicted_label = LABELS_INVERSE.get(predicted_index, f"Unknown (Index {predicted_index})")
        confidence = float(predictions[predicted_index]) # Convert to float for JSON serialization

        print(f"Predicted Animal: {predicted_label}, (Confidence: {confidence:.2f})")
        return jsonify({'label': predicted_label, 'confidence': confidence}), 200

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_resources() # Load model and labels when the script starts
    print("Starting Flask prediction service...")
    # Run on a different port than your Java backend (e.g., 5000)
    app.run(host='0.0.0.0', port=5000)