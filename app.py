from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
model = load_model("model.h5")  # Your model expects input shape=(5,)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

classes = ['Dog', 'Cat']

def extract_features(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0  # Normalize

    gray = np.mean(img_array, axis=2)  # Convert to grayscale
    features = [
        np.mean(gray),
        np.std(gray),
        np.max(gray),
        np.min(gray),
        gray[75, 75]  # center pixel
    ]
    return np.array([features])  # Shape (1, 5)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'image' not in request.files:
            return "No file uploaded.", 400

        file = request.files['image']
        if file.filename == '':
            return "No selected file.", 400

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        features = extract_features(file_path)
        prediction = model.predict(features)
        label = classes[int(prediction[0] > 0.5)]

        return render_template("index.html", prediction=label, image_path=file_path)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
