from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import io

app = Flask(__name__)

# Load the model
model_path = os.path.join('model', 'fire_detection_model.h5')
model = tf.keras.models.load_model(model_path)

def predict_single_image(image, img_size=64):
    # Convert PIL image to cv2 format
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Resize
    img = cv2.resize(img, (img_size, img_size))
    
    # Convert back to RGB for model input
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize
    img = img / 255.0
    
    # Expand dimensions to make it (1, img_size, img_size, 3)
    img_input = np.expand_dims(img, axis=0)
    
    # Predict
    pred = model.predict(img_input)
    pred_class = int(pred[0][0] > 0.5)  # 0 or 1
    confidence = float(pred[0][0]) * 100
    
    # Map to label
    label = "Fire" if pred_class == 1 else "No Fire"
    return label, confidence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Read image file
        image = Image.open(file)
        
        # Make prediction
        result, confidence = predict_single_image(image)
        
        return jsonify({
            'prediction': result,
            'confidence': f"{confidence:.2f}%"
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 