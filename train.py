import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import time

from feature_extractor import extract_features
from model_utils import (
    LogisticRegressionScratch,
    KNearestNeighborsScratch,
    LinearSVMScratch,
    accuracy,
    confusion_matrix,
    precision_recall_f1,
    save_model
)

# Dataset paths
TRAIN_DIR = 'dataset/ASL_Alphabet/asl_alphabet_train/asl_alphabet_train'
TEST_DIR = 'dataset/ASL_Alphabet/asl_alphabet_test/asl_alphabet_test'
IMG_LIMIT_PER_CLASS = 100

def load_data_from_dir(directory):
    X, y = [], []

    # Check if directory is flat (contains image files directly)
    is_flat = all(os.path.isfile(os.path.join(directory, f)) for f in os.listdir(directory))

    if is_flat:
        print("⚠️ Test directory is flat. Inferring label from filename.")
        for file in os.listdir(directory):
            img_path = os.path.join(directory, file)
            if not img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            label = file[0].upper()
            img = cv2.imread(img_path)
            if img is None:
                continue
            feat = extract_features(img)
            X.append(feat)
            y.append(label)
    else:
        for label in os.listdir(directory):
            label_path = os.path.join(directory, label)
            if not os.path.isdir(label_path):
                continue
            count = 0
            for file in os.listdir(label_path):
                if count >= IMG_LIMIT_PER_CLASS:
                    break
                img_path = os.path.join(label_path, file)
                if not os.path.isfile(img_path):
                    continue
                if not img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                img = cv2.imread(img_path)
                if img is None:
                    continue
                feat = extract_features(img)
                X.append(feat)
                y.append(label)
                count += 1

    return np.array(X), np.array(y)

# Load training data
print("Loading training data...")
X_train, y_train = load_data_from_dir(TRAIN_DIR)
print(f"Training samples: {len(X_train)}")

# Load testing data
print("Loading testing data...")
X_test, y_test = load_data_from_dir(TEST_DIR)
print(f"Testing samples: {len(X_test)}")

if len(X_train) == 0:
    raise ValueError("❌ No training data found. Please check TRAIN_DIR.")
if len(X_test) == 0:
    raise ValueError("❌ No testing data found. Please check TEST_DIR or use a valid format.")

# Label encoding
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)
num_classes = len(le.classes_)

# Train model
#print("\nTraining Logistic Regression from scratch...")
#model = LogisticRegressionScratch(lr=0.1, epochs=500, reg=0.01)

#print("\nTraining K Nearest Neighbors from scratch...")
#model = KNearestNeighborsScratch(k=3)

print("\nTraining Linear SVM from scratch...")
model = LinearSVMScratch(lr=0.01, epochs=300, reg=0.01)


start = time.time()
model.train(X_train, y_train_enc)
print(f"Training completed in {time.time() - start:.2f} seconds.\n")


# Evaluate
print("Evaluating model...")
y_pred = model.predict(X_test)

acc = accuracy(y_test_enc, y_pred)
cm = confusion_matrix(y_test_enc, y_pred, num_classes)
precision, recall, f1 = precision_recall_f1(cm)

print(f"Accuracy: {acc * 100:.2f}%")

# Plot Confusion Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues', fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# Plot Precision, Recall, F1 Score
plt.figure(figsize=(10, 5))
x = np.arange(len(le.classes_))
plt.plot(x, precision, label="Precision", marker='o')
plt.plot(x, recall, label="Recall", marker='s')
plt.plot(x, f1, label="F1-Score", marker='^')
plt.xticks(x, le.classes_, rotation=90)
plt.legend()
plt.grid(True)
plt.title("Precision, Recall, F1 Score per Class")
plt.tight_layout()
plt.savefig("metrics.png")
plt.show()

# Save model
save_model(model, le, 'model.pkl')
print("Model saved as model.pkl")
