# from_scratch_model/model_utils.py

import numpy as np
import pickle

# Logistic Regression from scratch
class LogisticRegressionScratch:
    def __init__(self, lr=0.01, epochs=1000, reg=0.01):
        self.lr = lr
        self.epochs = epochs
        self.reg = reg

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def train(self, X, y):
        self.classes = np.unique(y)
        self.weights = {}
        for cls in self.classes:
            y_binary = (y == cls).astype(int)
            w = np.zeros(X.shape[1])
            for _ in range(self.epochs):
                z = X @ w
                h = self._sigmoid(z)
                grad = X.T @ (h - y_binary) / X.shape[0] + self.reg * w
                w -= self.lr * grad
            self.weights[cls] = w

    def predict(self, X):
        scores = np.array([X @ self.weights[cls] for cls in self.classes]).T
        return self.classes[np.argmax(scores, axis=1)]

# kNN from scratch
class KNearestNeighborsScratch:
    def __init__(self, k=3):
        self.k = k

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        preds = []
        for x in X:
            dists = np.linalg.norm(self.X_train - x, axis=1)
            k_idx = np.argsort(dists)[:self.k]
            labels, counts = np.unique(self.y_train[k_idx], return_counts=True)
            preds.append(labels[np.argmax(counts)])
        return np.array(preds)
    
# Linear SVM from scratch (using SGD)
class LinearSVMScratch:
    def __init__(self, lr=0.01, epochs=1000, reg=0.01):
        self.lr = lr
        self.epochs = epochs
        self.reg = reg

    def train(self, X, y):
        self.classes = np.unique(y)
        self.weights = {}
        for cls in self.classes:
            y_binary = np.where(y == cls, 1, -1)
            w = np.zeros(X.shape[1])
            for _ in range(self.epochs):
                for i in range(len(X)):
                    if y_binary[i] * (X[i] @ w) < 1:
                        w += self.lr * (y_binary[i] * X[i] - self.reg * w)
                    else:
                        w -= self.lr * self.reg * w
            self.weights[cls] = w

    def predict(self, X):
        scores = np.array([X @ self.weights[cls] for cls in self.classes]).T
        return self.classes[np.argmax(scores, axis=1)]


# Evaluation Metrics
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm

def precision_recall_f1(cm):
    precisions, recalls, f1s = [], [], []
    for i in range(len(cm)):
        TP = cm[i, i]
        FP = sum(cm[:, i]) - TP
        FN = sum(cm[i, :]) - TP
        precision = TP / (TP + FP + 1e-9)
        recall = TP / (TP + FN + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return precisions, recalls, f1s

# Save/Load model
def save_model(model, labels, filename='model.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump((model, labels), f)

def load_model(filename='model.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)
