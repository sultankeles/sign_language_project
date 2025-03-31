# Sign Language Number Recognition

This project aims to recognize hand gestures representing numbers (0 to 9) in real-time using a Convolutional Neural Network (CNN) trained on a custom dataset of grayscale hand images.

---

## 📦 Download Dataset

You can download the full training dataset (~175,000 images) from the link below:

👉 [Download merged_rotated.zip from Google Drive](https://drive.google.com/drive/folders/1zn-vFtKoGX8axPeU9McAr1j4z0t8KWJ5?usp=drive_link)

> Unzip the folder and place it in the project root before training.


## 📁 Project Structure

```bash
├── merged_dataset/           # Original + manually added images
├── merged_rotated/           # Rotated/augmented dataset (final training set)
├── train_model.py            # Model training script
├── realtime_prediction       # Real-time prediction (with smoothing)
├── confusion_matrix_only.py  # Evaluate model performance
├── best_model.keras          # Saved trained model
├── README.md                 # Project documentation (this file)
```

---

## 📊 Dataset
- Includes 10 classes: digits `0` to `9`
- Grayscale images, size: **64x64 pixels**
- Augmented with left/right rotations to improve generalization
- Total images: ~175,000

---

## 🧠 Model Architecture

- 2x Conv2D + MaxPooling
- Flatten + Dense (256 units)
- Dropout (0.5)
- Output layer: 10 classes (softmax)

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```

---

## 🚀 Training

Run the training script:
```bash
python train_model.py
```
This will:
- Load and preprocess data
- Split into training/validation sets (80/20)
- Train the model for up to 20 epochs
- Save the best model as `best_model.keras`

---

## 🎥 Real-Time Prediction

Use your webcam to detect hand signs:
```bash
python live_predict_smoothed.py
```
Features:
- ROI-based detection
- Confidence threshold: `0.80`
- Voting system for stability (2-frame history)

---

## 📊 Model Evaluation

Generate confusion matrix:
```bash
python confusion_matrix_only.py
```
This will display a color-coded matrix showing class-wise performance on the validation set.

---

## ✅ Requirements
- Python 3.8+
- OpenCV
- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-learn

---

## 📌 Credits
Developed by **Orhan AYDIN** **Sultan KELES** and **Rakesh NEGI** as part of MSc AI & Computer Vision coursework.

Inspired by real-world gesture recognition needs and optimized for robustness in live settings.

---
