from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import numpy as np
import os

app = Flask(__name__)

# Load model once when server starts
model = load_model(
    "saved_models/chest_xray_model.h5"
)

classes = [
    "COVID 19",
    "HEALTHY",
    "PNEUMONIA"
]


def preprocess_image(img_path):

    img = image.load_img(
        img_path,
        target_size=(150, 150)
    )

    img_array = image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None

    if request.method == "POST":

        uploaded_file = request.files["image"]

        filepath = os.path.join(
            "uploads",
            uploaded_file.filename
        )

        uploaded_file.save(filepath)

        processed_image = preprocess_image(
            filepath
        )

        pred = model.predict(
            processed_image
        )

        print("\nRaw Prediction:")
        print(pred)

        class_index = np.argmax(pred)

        prediction = classes[class_index]

        confidence = round(
            float(np.max(pred)) * 100,
            2
        )

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )


@app.route("/predict", methods=["POST"])
def predict_api():

    uploaded_file = request.files["image"]

    filepath = os.path.join(
        "uploads",
        uploaded_file.filename
    )

    uploaded_file.save(filepath)

    processed_image = preprocess_image(
        filepath
    )

    pred = model.predict(
        processed_image
    )

    class_index = np.argmax(pred)

    prediction = classes[class_index]

    confidence = round(
        float(np.max(pred)) * 100,
        2
    )

    return jsonify({
        "prediction": prediction,
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )