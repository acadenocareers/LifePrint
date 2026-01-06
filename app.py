# --------------------------------------------------
# Force TensorFlow to CPU (IMPORTANT for Render)
# --------------------------------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
app = Flask(__name__)

# Strong CORS for Flutter Web + Mobile
CORS(app, resources={r"/*": {"origins": "*"}})

# --------------------------------------------------
# Load Face Detector (OpenCV DNN)
# --------------------------------------------------
FACE_PROTO = "face_detector/deploy.prototxt"
FACE_MODEL = "face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel"

if not os.path.exists(FACE_PROTO) or not os.path.exists(FACE_MODEL):
    raise FileNotFoundError("Face detector files not found")

face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)

# --------------------------------------------------
# Load Emotion Detection Model
# --------------------------------------------------
MODEL_PATH = "model/emotion_model.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Emotion model not found")

emotion_model = load_model(MODEL_PATH)

emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# --------------------------------------------------
# Health Check Route (REQUIRED for Render)
# --------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "service": "LifePrint Emotion Detection API"
    })

# --------------------------------------------------
# HTML Test Page (Optional)
# --------------------------------------------------
@app.route("/test", methods=["GET"])
def test_page():
    return send_from_directory(".", "index.html")

# --------------------------------------------------
# Emotion Prediction Route
# --------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        file = request.files["image"]

        # Read image
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        h, w = img.shape[:2]

        # Face detection
        blob = cv2.dnn.blobFromImage(
            img, 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        face_net.setInput(blob)
        detections = face_net.forward()

        results = []

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                face = img[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                # Preprocess face
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

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------------------------------------
# OPTIONS handler (Flutter Web preflight)
# --------------------------------------------------
@app.route("/predict", methods=["OPTIONS"])
def predict_options():
    return jsonify({}), 200

# --------------------------------------------------
# App Runner (Local only – Render uses Gunicorn)
# --------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
