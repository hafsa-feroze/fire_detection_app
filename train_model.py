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

def train_model():
 

    df = pd.read_csv('trainLabels.csv') 
    image_folder = 'training/'
    img_size = 64 

    X = []
    y = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_path = image_folder + row['img']
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        img = img / 255.0  # Normalize to 0-1
        X.append(img)
        y.append(row['label'])

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    print(X.shape, y.shape) 

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.shape, X_val.shape)

    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(X_train, y_train, epochs=6, validation_data=(X_val, y_val))

    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {val_acc}")
    print(f"Validation loss: {val_loss}")
    model.save('fire_detection_model.h5')

    # Predict
    preds = model.predict(X_val)
    preds = (preds > 0.5).astype(int).flatten()  # Convert probabilities to 0/1 and flatten to 1D

    cm = confusion_matrix(y_val, preds)
    print(cm)

    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Fire', 'Fire'], yticklabels=['No Fire', 'Fire'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix Heatmap')
    plt.show()
    return model