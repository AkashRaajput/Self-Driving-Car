# Render Deployment Guide

## What This App Deploys

This is a Flask web service that loads `model.h5`, accepts an uploaded driving video, runs frame-by-frame TensorFlow inference, and renders synchronized dashboard telemetry.

## Verified Locally

- `model.h5` loads with TensorFlow.
- Model input shape: `(None, 66, 200, 3)`.
- Model output shape: `(None, 1)`.
- Preprocessing in `app.py` resizes frames to `(200, 66)` and normalizes to `0..1`.
- OpenCV is required for video decoding.
- Use `opencv-python-headless` on Render instead of GUI OpenCV.

## Required Files

- `app.py`
- `model.h5`
- `requirements.txt`
- `Procfile`
- `runtime.txt`
- `templates/`
- `static/images/steering-wheel.png`

## Render Settings

1. Push this project to GitHub.
2. In Render, create a new **Web Service**.
3. Connect the GitHub repository.
4. Use:
   - Language: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 300`
5. Optional environment variables:
   - `MODEL_PATH=model.h5`
   - `MAX_CONTENT_LENGTH=209715200`
6. Deploy.
7. Open `/healthz` after deployment to verify the service is up and the model is loaded.

## Notes

Render requires web services to bind to `0.0.0.0` and the `$PORT` environment variable. Gunicorn handles that in production.

The uploaded videos and generated runtime files are temporary. Render's normal filesystem is ephemeral, so do not rely on uploads remaining after restarts or redeploys.

If the deploy exceeds free-tier memory, use a paid instance type or reduce video size before uploading.
