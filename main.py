
import os

# Set resource limits before importing TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
IMAGE_SIZE = 128

# Load the model once when the application starts
model = load_model("model.h5", compile=False)

class_labels = ["pituitary", "glioma", "notumor", "meningioma"]


def predict_tumor(image_path):
    img = load_img(
        image_path,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        color_mode="rgb"
    )
    img_array = img_to_array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    predicted_class_index = int(np.argmax(predictions[0]))
    confidence_score = float(np.max(predictions[0]))

    if class_labels[predicted_class_index] == "notumor":
        result = "No Tumor"
    else:
        result = f"Tumor: {class_labels[predicted_class_index]}"

    return result, confidence_score


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")

        if not file or not file.filename:
            return render_template(
                "index.html",
                result="Please select an image.",
                confidence=None
            ), 400

        filename = secure_filename(file.filename)
        extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        if extension not in ALLOWED_EXTENSIONS:
            return render_template(
                "index.html",
                result="Please upload a JPG, JPEG, or PNG image.",
                confidence=None
            ), 400

        file_location = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_location)

        try:
            result, confidence = predict_tumor(file_location)
            return render_template(
                "index.html",
                result=result,
                confidence=f"{confidence * 100:.2f}%",
                file_path=f"/uploads/{filename}"
            )
        except Exception:
            app.logger.exception("Image prediction failed")
            return render_template(
                "index.html",
                result="Prediction failed. Please try another image.",
                confidence=None
            ), 500

    return render_template("index.html", result=None, confidence=None)


@app.route("/uploads/<path:filename>")
def get_uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
