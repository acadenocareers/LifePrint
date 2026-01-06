# --------------------------------------------------
# Force TensorFlow to CPU (IMPORTANT for Render)
# --------------------------------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# --------------------------------------------------
# Imports
# --------------------------------------------------
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

# --------------------------------------------------
# App configuration
# --------------------------------------------------
app = Flask(__name__, static_folder=".")

CORS(app, resources={r"/*": {"origins": "*"}})

# --------------------------------------------------
# Load Face Detector
# --------------------------------------------------
FACE_PROTO = "face_detector/deploy.prototxt"
FACE_MODEL = "face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel"

face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)

# --------------------------------------------------
# Load Emotion Model
# --------------------------------------------------
MODEL_PATH = "model/emotion_model.h5"
emotion_model = load_model(MODEL_PATH)

emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# --------------------------------------------------
# Health Check
# --------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "LifePrint Emotion Detection API",
        "status": "running",
        "usage": {
            "POST /predict": "Send image as multipart/form-data",
            "GET /test": "HTML test page"
        }
    })

# --------------------------------------------------
# HTML Test Page
# --------------------------------------------------
@app.route("/test", methods=["GET"])
def test_page():
    return send_from_directory(".", "index.html")

# --------------------------------------------------
# Predict (POST – real API)
# --------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Image field missing"}), 400

    file = request.files["image"]

    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104, 177, 123))
    face_net.setInput(blob)
    detections = face_net.forward()

    results = []

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face = img[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (48, 48))
            face = face / 255.0
            face = face.reshape(1, 48, 48, 1)

            preds = emotion_model.predict(face, verbose=0)
            emotion = emotions[int(np.argmax(preds))]

            results.append({
                "emotion": emotion,
                "confidence": round(confidence, 3)
            })

    if not results:
        return jsonify({"message": "No face detected"}), 200

    return jsonify(results), 200

# --------------------------------------------------
# Predict (GET – browser-friendly message)
# --------------------------------------------------
@app.route("/predict", methods=["GET"])
def predict_help():
    return jsonify({
        "message": "This endpoint only accepts POST requests",
        "how_to_use": {
            "method": "POST",
            "content_type": "multipart/form-data",
            "field": "image"
        }
    }), 200

# --------------------------------------------------
# OPTIONS (Flutter Web preflight)
# --------------------------------------------------
@app.route("/predict", methods=["OPTIONS"])
def predict_options():
    return jsonify({}), 200

# --------------------------------------------------
# Local run (Render uses Gunicorn)
# --------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
