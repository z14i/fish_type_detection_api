from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load the model
model_path = os.path.expanduser("./fish_classification_model.h5")  # Adjusted for relative path
model = load_model(model_path)

# Define class names based on the model's output
class_names = ['Black Sea Sprat', 'Gilt-Head Bream', 'Hourse Mackerel', 
               'Red Mullet', 'Red Sea Bream', 'Sea Bass', 'Shrimp', 
               'Striped Red Mullet', 'Trout']

def prepare_image(image, target_size):
    """Preprocess the uploaded image for prediction."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/', methods=['GET'])
def hello():
    """Root endpoint for health check."""
    return jsonify({"message": "API is working"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Open the image
        image = Image.open(file)
        
        # Preprocess the image
        processed_image = prepare_image(image, target_size=(224, 224))
        
        # Perform prediction
        prediction = model.predict(processed_image)
        
        # Get the class with the highest probability
        predicted_class = class_names[np.argmax(prediction[0])]
        
        return jsonify({"predicted_class": predicted_class})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Expose WSGI app as a Vercel handler
def handler(request, *args):
    """Handler function for Vercel serverless environment."""
    return app(request.environ, start_response)

if __name__ == "__main__":
    app.run(debug=True)
