import gc
import logging
import os
import uuid
from pathlib import Path

# Keep TensorFlow conservative on small Render instances. These must be set
# before TensorFlow is imported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename


logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        int(os.environ.get("TF_NUM_INTRAOP_THREADS", "1"))
    )
    tf.config.threading.set_inter_op_parallelism_threads(
        int(os.environ.get("TF_NUM_INTEROP_THREADS", "1"))
    )
except RuntimeError:
    logger.warning("TensorFlow thread settings were already initialized.")


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / os.environ.get("MODEL_PATH", "model.h5")
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"

MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", 75 * 1024 * 1024))
MAX_VIDEO_SECONDS = float(os.environ.get("MAX_VIDEO_SECONDS", "30"))
FRAME_SAMPLE_INTERVAL = max(1, int(os.environ.get("FRAME_SAMPLE_INTERVAL", "10")))
MAX_SAMPLED_FRAMES = max(1, int(os.environ.get("MAX_SAMPLED_FRAMES", "120")))
INFERENCE_BATCH_SIZE = max(1, int(os.environ.get("INFERENCE_BATCH_SIZE", "8")))
WORKING_FRAME_WIDTH = max(200, int(os.environ.get("WORKING_FRAME_WIDTH", "320")))
WORKING_FRAME_HEIGHT = max(160, int(os.environ.get("WORKING_FRAME_HEIGHT", "160")))
ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm", "mpeg", "mpg"}


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

logger.info("Loading model from %s", MODEL_PATH)
model = load_model(MODEL_PATH, compile=False)
logger.info("Model loaded successfully")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def upload_limit_mb():
    return round(MAX_CONTENT_LENGTH / (1024 * 1024))


def preprocess_frame(frame_bgr):
    """Match the training pipeline while shrinking high-resolution uploads early."""
    frame = cv2.resize(
        frame_bgr,
        (WORKING_FRAME_WIDTH, WORKING_FRAME_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame[60:135, :, :]
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    frame = cv2.resize(frame, (200, 66), interpolation=cv2.INTER_AREA)
    frame = frame.astype(np.float32) / 255.0
    return frame


def predict_batch(batch, angles):
    if not batch:
        return

    inputs = np.asarray(batch, dtype=np.float32)
    predictions = model.predict(inputs, batch_size=len(inputs), verbose=0)
    angles.extend(round(float(value), 6) for value in predictions.reshape(-1))

    del inputs, predictions
    batch.clear()


def process_video(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError("Could not read uploaded video")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps <= 0:
            fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        detected_duration = total_frames / fps if total_frames > 0 else 0
        frames_to_scan = total_frames

        if total_frames > 0 and detected_duration > MAX_VIDEO_SECONDS:
            frames_to_scan = int(MAX_VIDEO_SECONDS * fps)
            logger.info(
                "Trimming processing window from %.2fs to %.2fs",
                detected_duration,
                MAX_VIDEO_SECONDS,
            )

        angles = []
        batch = []
        frame_index = 0
        sampled_frames = 0

        logger.info(
            "Processing %s at %.2f fps, sample_every=%s, max_samples=%s",
            video_path.name,
            fps,
            FRAME_SAMPLE_INTERVAL,
            MAX_SAMPLED_FRAMES,
        )

        while sampled_frames < MAX_SAMPLED_FRAMES:
            if frames_to_scan and frame_index >= frames_to_scan:
                break

            if frame_index % FRAME_SAMPLE_INTERVAL == 0:
                ok, frame = cap.read()
                if not ok:
                    break

                batch.append(preprocess_frame(frame))
                sampled_frames += 1

                if len(batch) >= INFERENCE_BATCH_SIZE:
                    predict_batch(batch, angles)
            else:
                ok = cap.grab()
                if not ok:
                    break

            frame_index += 1

            if frame_index % 150 == 0:
                logger.info("Scanned %s frames, sampled %s", frame_index, sampled_frames)

        predict_batch(batch, angles)

        if not angles:
            raise ValueError("No usable frames found in uploaded video")

        effective_fps = fps / FRAME_SAMPLE_INTERVAL

        return {
            "angles": angles,
            "first_angle": round(angles[0], 4),
            "source_fps": round(float(fps), 3),
            "effective_fps": round(float(effective_fps), 3),
            "scanned_frames": frame_index,
            "sampled_frames": sampled_frames,
            "duration_seconds": round(frame_index / fps, 3),
        }
    finally:
        cap.release()
        gc.collect()


@app.route("/")
def home():
    return render_template(
        "index.html",
        max_upload_mb=upload_limit_mb(),
        max_video_seconds=int(MAX_VIDEO_SECONDS),
    )


@app.errorhandler(RequestEntityTooLarge)
def handle_large_upload(error):
    return render_template(
        "index.html",
        max_upload_mb=upload_limit_mb(),
        max_video_seconds=int(MAX_VIDEO_SECONDS),
        upload_error=(
            f"Request Entity Too Large. Please upload a video up to "
            f"{upload_limit_mb()} MB and {int(MAX_VIDEO_SECONDS)} seconds."
        ),
    ), 413


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "frame_sample_interval": FRAME_SAMPLE_INTERVAL,
        "max_video_seconds": MAX_VIDEO_SECONDS,
        "max_sampled_frames": MAX_SAMPLED_FRAMES,
    }


@app.route("/predict_video", methods=["POST"])
def predict_video():
    if "video" not in request.files:
        return "No video uploaded", 400

    video = request.files["video"]

    if not video.filename:
        return "No video selected", 400

    if not allowed_file(video.filename):
        return "Unsupported video format", 400

    filename = secure_filename(video.filename)
    unique_filename = f"{uuid.uuid4().hex}_{filename}"
    video_path = UPLOAD_FOLDER / unique_filename

    try:
        video.save(video_path)
        result = process_video(video_path)
    except ValueError as exc:
        if video_path.exists():
            video_path.unlink(missing_ok=True)
        logger.exception("Video processing failed")
        return str(exc), 400
    except Exception:
        if video_path.exists():
            video_path.unlink(missing_ok=True)
        logger.exception("Unexpected video processing error")
        return "Video processing failed. Try a shorter 10-30 second clip.", 500

    logger.info(
        "Finished %s: scanned=%s sampled=%s effective_fps=%.3f",
        unique_filename,
        result["scanned_frames"],
        result["sampled_frames"],
        result["effective_fps"],
    )

    return render_template(
        "result.html",
        video_path=url_for("static", filename=f"uploads/{unique_filename}"),
        angle=result["first_angle"],
        angles=result["angles"],
        fps=result["effective_fps"],
        source_fps=result["source_fps"],
        sampled_frames=result["sampled_frames"],
        scanned_frames=result["scanned_frames"],
        duration_seconds=result["duration_seconds"],
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
