from flask import Flask, request, jsonify
import os
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from flask_cors import CORS

app = Flask(__name__)

CORS(app)
CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}})


model = load_model("model")

directory_path = "flowers"
items = os.listdir(directory_path)
folder_names = []
for item in items:
    item_path = os.path.join(directory_path, item)
    if os.path.isdir(item_path):
        folder_names.append(item)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]

        if file:
            file_path = "uploaded_image.jpg"
            file.save(file_path)
            image = load_img(file_path, target_size=(150, 150))
            image = img_to_array(image)
            image = image / 255.0
            image = image.reshape((1,) + image.shape)

            predictions = model.predict(image)

            class_index = predictions.argmax()
            result = folder_names[class_index]

            return jsonify({"result": result})

        return jsonify({"error": "No file provided"})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
