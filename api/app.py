from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

model_path = os.path.expanduser("./fish_classification_model.h5")
model = load_model(model_path)

class_names = ['Black Sea Sprat', 'Gilt-Head Bream', 'Hourse Mackerel', 
               'Red Mullet', 'Red Sea Bream', 'Sea Bass', 'Shrimp', 
               'Striped Red Mullet', 'Trout'] 

def prepare_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/', methods=['GET'])
def hello():
    return "hi"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    try:
        image = Image.open(file)
        processed_image = prepare_image(image, target_size=(224, 224))
        prediction = model.predict(processed_image)
        predicted_class = class_names[np.argmax(prediction[0])]
        return jsonify({"predicted_class": predicted_class})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Expose WSGI app as a Vercel handler
def handler(event, context):
    return app(event, context)
