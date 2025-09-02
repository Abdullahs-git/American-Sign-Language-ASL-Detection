# 🖐 American Sign Language (ASL) Detection

This project detects and classifies ASL alphabets using **classical ML models**: Support Vector Machine (SVM), Logistic Regression (LR), and k-Nearest Neighbors (k-NN). Includes handcrafted features (HOG, LBP, Color Histogram) and pre-trained model weights.

## 🚀 Features

1. Real-time ASL detection via webcam
2. Handcrafted feature extraction: HOG, LBP, Color Histogram
3. Comparison of models: SVM, Logistic Regression, k-NN
4. Pre-trained model weights included (`.pkl`)
5. Evaluation metrics: Accuracy, Precision, Recall, F1-score
6. Confusion matrix and metric visualization

## 🛠️ Installation

1. Clone the repository
   ```bash
   git clone <your-repo-link>
   cd ASL-Detection-ML
   ```
2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run real-time detection

```bash
python realtime_predict.py
```

4. Train or evaluate models

```bash
python train.py
python evaluate.py
```

## 🧪 Models & Weights

- **SVM** → `weights/svm_model.pkl`
- **Logistic Regression** → `weights/lr_model.pkl`
- **k-NN** → `weights/knn_model.pkl`

Models can be loaded directly for predictions without retraining.

## 📊 Evaluation

- Accuracy, Precision, Recall, F1-score for all models
- Confusion matrix visualization
- Class-wise metrics plotted for comparison

## 🔒 License

This project is licensed under the **MIT License** – free to use, modify, and distribute with attribution.
