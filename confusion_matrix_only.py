import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# 📁 Dataset settings
dataset_dir = "merged_rotated"
img_size = 64
num_classes = 10

# 🔄 Load images and labels
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
        if img is not None:
            img = cv2.resize(img, (img_size, img_size))
            X.append(img)
            Y.append(int(label))

X = np.array(X).reshape(-1, img_size, img_size, 1) / 255.0
Y = to_categorical(Y, num_classes)

print(f"✅ Total images loaded: {X.shape[0]}")

# 🔀 Create validation set
_, X_val, _, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# 🧠 Load trained model
model = load_model("best_model.keras")
print("✅ Model loaded.")

# 🔮 Predict
y_pred = model.predict(X_val, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(Y_val, axis=1)

# 📊 Confusion Matrix
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Validation Set")
plt.show()
