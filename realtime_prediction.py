import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# Load trained model
model = load_model("best_model.keras")
label_map = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"
}

# Configuration
IMG_SIZE = 64
CONFIDENCE_THRESHOLD = 0.80
VOTE_HISTORY = 2
history = deque(maxlen=VOTE_HISTORY)

# Image preprocessing
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    return normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Define ROI (Region of Interest)
    x, y, w, h = 220, 100, 200, 200
    roi = frame[y:y+h, x:x+w]
    processed = preprocess(roi)

    # Make prediction
    prediction = model.predict(processed, verbose=0)
    class_id = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    # Voting (use last N predictions to smooth result)
    if confidence > CONFIDENCE_THRESHOLD:
        history.append(class_id)
        final_pred = max(set(history), key=history.count)
        text = f"{label_map[final_pred]} ({confidence*100:.1f}%)"
        color = (0, 255, 0)
    else:
        text = "?"
        color = (0, 0, 255)

    # Display ROI and prediction result
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Live Prediction (Smoothed)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
