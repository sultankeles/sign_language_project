# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 15:57:16 2025
@author: Orhan
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Settings
dataset_dir = "merged_rotated"
img_size = 64
num_classes = 10  # digits 0–9

# Folder name → label name map (for reference only)
label_map = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"
}

# Load images and labels
X, Y = [], []

for label in os.listdir(dataset_dir):
    path = os.path.join(dataset_dir, label)
    if not os.path.isdir(path):
        continue

    for fname in os.listdir(path):
        if not fname.endswith(".jpg"):
            continue
        img_path = os.path.join(path, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (img_size, img_size))
        X.append(img)
        Y.append(int(label))  # folder name → label (0-9)

X = np.array(X).reshape(-1, img_size, img_size, 1) / 255.0
Y = to_categorical(Y, num_classes=num_classes)

print(f"✅ Data loaded: {X.shape[0]} images")

# Split into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
print(f"🧪 Training set: {X_train.shape[0]}, Validation set: {X_val.shape[0]}")

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Training callbacks
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint("best_model.keras", save_best_only=True, verbose=1)
]

#  Train the model
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=20,
    batch_size=256,
    callbacks=callbacks
)

print("✅ Model trained successfully and saved as 'best_model.keras'")
