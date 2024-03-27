from flask import Flask, request, jsonify
import io
import base64
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.models import load_model

app = Flask(name)

# Load the trained model
model = load_model('final_project.h5')

# Function to predict the class of an uploaded image
def img_pred(uploaded_image):
    img = Image.open(io.BytesIO(uploaded_image))
    opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(opencvImage, (150, 150))
    img = img.reshape(1, 150, 150, 3)

    p = model.predict(img)
    p = np.argmax(p, axis=1)[0]

    class_names = ['Glioma Tumor', 'No Tumor', 'Meningioma Tumor', 'Pituitary Tumor']
    prediction = class_names[p]

    return prediction

# Create the endpoint for the model
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'})
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'})
        if file:
            img = file.read()
            prediction = img_pred(img)
            return jsonify({'prediction': prediction})
        else:
            return jsonify({'error': 'No image uploaded'})

if name == 'main':
    app.run(debug=True)
