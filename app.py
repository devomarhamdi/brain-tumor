from flask import Flask, request, jsonify
import io
import base64
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)

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

@app.route('/')
def index():
    return 'hello world'

# Create the endpoint for the model
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            img = file.read()
            # prediction = img_pred(img)
            return jsonify({'prediction': img})
        else:
            return jsonify({'error': 'No image uploaded'})

if __name__ == '__main__':
    app.run(debug=True)