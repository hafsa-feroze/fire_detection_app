# Fire Detection Web Application

This is a Flask web application that uses a TensorFlow model to detect fire in uploaded images.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Make sure your model file `fire_detection_model.h5` is in the `model` directory.

2. Run the Flask application:
```bash
python app.py
```

3. Open your web browser and navigate to `http://localhost:5000`

## Usage

1. Click the "Choose File" button to select an image
2. Click "Detect Fire" to analyze the image
3. The prediction result will be displayed below the form

## Features

- Image upload functionality
- Real-time fire detection
- Confidence score display
- Responsive Bootstrap UI
- Loading indicators
- Error handling 