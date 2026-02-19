from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import base64
from PIL import Image
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load model
model = load_model("skintone_model.h5")   # rename to your model name

# Class names
CLASS_NAMES = ['Black', 'Brown', 'White']

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    
    return CLASS_NAMES[class_index]

@app.route("/", methods=["GET"])
def index():
    return render_template("test.html", prediction=None)

@app.route("/predict", methods=["POST"])
def upload_predict():

    # Case 1: Uploaded Image
    if "img" in request.files and request.files["img"].filename != "":
        file = request.files["img"]
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        result = predict_image(filepath)
        return render_template("test.html", prediction=result, img_path=filepath)

    # Case 2: Camera Image → Base64
    if "camera_image" in request.form and request.form["camera_image"] != "":
        data_url = request.form["camera_image"]

        header, encoded = data_url.split(",", 1)
        decoded = base64.b64decode(encoded)

        img = Image.open(BytesIO(decoded))
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], "camera_capture.png")
        img.save(filepath)

        result = predict_image(filepath)
        return render_template("test.html", prediction=result, img_path=filepath)

    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
