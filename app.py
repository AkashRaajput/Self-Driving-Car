from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)

model = load_model("model.h5", compile=False)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    image = Image.open(file).convert("RGB")
    image = image.resize((200, 66))

    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)

    return f"Predicted Steering Angle: {prediction[0][0]:.4f}"

if __name__ == "__main__":
    app.run(debug=True)