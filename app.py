from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

# ✅ FIX 1: correct __name__
app = Flask(__name__)
CORS(app)

# ✅ Load face detector
face_net = cv2.dnn.readNetFromCaffe(
    "face_detector/deploy.prototxt",
    "face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

# ✅ Load emotion model
emotion_model = load_model("model/emotion_model.h5")

emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


# ✅ Health check / home route (VERY IMPORTANT for Render)
@app.route("/")
def home():
    return "LifePrint API is running"


# ✅ Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104, 177, 123))
    face_net.setInput(blob)
    detections = face_net.forward()

    results = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")

            face = img[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (48, 48))
            face = face / 255.0
            face = face.reshape(1, 48, 48, 1)

            preds = emotion_model.predict(face, verbose=0)
            emotion = emotions[np.argmax(preds)]

            results.append({
                "emotion": emotion,
                "confidence": float(confidence)
            })

    return jsonify(results)


# ✅ FIX 2: correct __name__ check
if __name__ == "__main__":
    app.run()
