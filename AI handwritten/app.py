# app.py
from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import numpy as np

app = Flask(__name)

# Load the character recognition model
character_model = load_model('handwritten_character_recognition_model.h5')

# Load the sentence recognition model (if available)
# sentence_model = load_model('handwritten_sentence_recognition_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/recognize_character', methods=['POST'])
def recognize_character():
    image = request.files['image']
    # Preprocess the image and use the character recognition model
    # result = character_model.predict(preprocessed_image)
    # Return the result as a response

@app.route('/recognize_sentence', methods=['POST'])
def recognize_sentence():
    image = request.files['image']

    def preprocess_image(image_path, target_size):
    # Load the image
    image = Image.open(image_path)

    # Resize the image to the target size
    image = image.resize(target_size)

    # Normalize pixel values (assuming the model expects values in the range [0, 1])
    image = np.array(image) / 255.0

    # Convert the image to the expected data type (e.g., float32)
    image = image.astype(np.float32)

    # Expand dimensions to match the model's input shape if needed
    if len(image.shape) == 2:  # Convert grayscale image to 3 channels
        image = np.stack((image,) * 3, axis=-1)

    # Apply any additional model-specific preprocessing if required
    # Example: mean subtraction, channel reordering, etc.

    return image





