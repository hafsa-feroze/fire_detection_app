import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import cv2
from tqdm import tqdm  # progress bar
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image

def predict_single_image(img_path, model, img_size=64):
   

    # Load the image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load {img_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize
    img = cv2.resize(img, (img_size, img_size))

    # Normalize
    img = img / 255.0

    # Expand dimensions to make it (1, img_size, img_size, 3)
    img_input = np.expand_dims(img, axis=0)

    # Predict
    pred = model.predict(img_input)
    pred_class = int(pred[0][0] > 0.5)  # 0 or 1

    # Map to label
    label = "Fire" if pred_class == 1 else "No Fire"
    return label


# Load your Excel/CSV file
 # If you saved as Excel
# Or if CSV:
# df = pd.read_csv('/kaggle/input/your_csv_file.csv')


if __name__ == "__main__":

    from tensorflow.keras.models import load_model

    model = load_model('fire_detection_model.h5')

    #test_df = pd.read_csv('test.csv')

    # Path to the folder containing the actual test images
    test_image_folder = 'testing/'

    # Assume you've already trained your model and have it loaded as `model`
    # For example: model = ... (load or train the model here)

    # Loop over first 10 images
    # for idx, row in test_df.head(10).iterrows():
    #     img_path = test_image_folder + row['img']  # assuming the column is named 'img'
    img_path = 'testing/NON_FIRE (2511).jpg'
    img = Image.open(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    label = predict_single_image(img_path, model)
    print(f"Predicted Label = {label}")


