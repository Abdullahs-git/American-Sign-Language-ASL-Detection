# realtime_predict.py

import cv2
import numpy as np
from feature_extractor import extract_features
from model_utils import load_model

IMG_SIZE = 64

print("Loading model...")
model, label_encoder = load_model('KNNmodel.pkl')
print("Model loaded. Starting webcam...")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    roi = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    features = extract_features(roi).reshape(1, -1)

    pred = model.predict(features)[0]
    label = label_encoder.inverse_transform([pred])[0]

    cv2.putText(frame, f"Prediction: {label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
    cv2.imshow("ASL Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
