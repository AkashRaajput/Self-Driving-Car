from pathlib import Path
from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import os
import uuid

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(
    os.environ.get("MAX_CONTENT_LENGTH", 200 * 1024 * 1024)
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / os.environ.get("MODEL_PATH", "model.h5")

# Folders
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
ANGLES_FOLDER = BASE_DIR / "static" / "data"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# Load trained model
model = load_model(MODEL_PATH, compile=False)

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
ANGLES_FOLDER.mkdir(parents=True, exist_ok=True)


# Same preprocessing used during training
def img_preprocess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img


@app.route("/")
def home():
    return render_template("index.html")


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_loaded": model is not None}


@app.route("/predict_video", methods=["POST"])
def predict_video():
    if "video" not in request.files:
        return "No video uploaded"

    video = request.files["video"]

    if video.filename == "":
        return "No video selected"

    filename = secure_filename(video.filename)
    unique_filename = f"{uuid.uuid4().hex}_{filename}"
    video_path = UPLOAD_FOLDER / unique_filename
    video.save(video_path)

    # Read video
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return "Could not read uploaded video", 400

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 30

    angles = []
    frame_count = 0

    print("Starting video processing...")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed = img_preprocess(frame)
        processed = np.array([processed])

        angle = float(model.predict(processed, verbose=0)[0][0])
        angles.append(angle)

        frame_count += 1

        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames")

    cap.release()

    print(f"Finished. Saved {len(angles)} angles.")

    first_angle = 0

    if len(angles) > 0:
        first_angle = round(angles[0], 4)

    return render_template(
        "result.html",
        video_path=url_for("static", filename=f"uploads/{unique_filename}"),
        angle=first_angle,
        angles=angles,
        fps=fps,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
