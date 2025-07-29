# mnist-pytorch-classifier

# 🧠 MNIST PyTorch Classifier

A beginner-friendly project to build, train, and evaluate a neural network using **PyTorch** for classifying handwritten digits from the **MNIST dataset**.

This project is designed as a step-by-step practical to help understand the complete PyTorch workflow: from data loading and model creation to training, evaluation, and generating Kaggle submissions.

---

## 📌 Project Objective

We aim to:
- Understand how feedforward neural networks work
- Build a custom image classification model using PyTorch
- Train it on the MNIST digit dataset
- Evaluate performance using accuracy and classification metrics
- Generate predictions for Kaggle’s `digit-recognizer` challenge

---

## 🗂️ Dataset

The project uses the [Kaggle Digit Recognizer Dataset](https://www.kaggle.com/competitions/digit-recognizer), which includes:
- `train.csv`: 42,000 labeled grayscale images (28x28)
- `test.csv`: 28,000 unlabeled images (used for submission)

---

## 🛠️ Tools & Libraries

- Python 3.x
- PyTorch
- Torchvision
- Pandas, NumPy
- Matplotlib (for plots)
- Scikit-learn (for classification metrics)

---

## 🏗️ Model Architecture

A simple **feedforward neural network** with:
- Input layer: 784 nodes (flattened 28x28 image)
- Hidden layers: 128 → 64 neurons with ReLU activations
- Output layer: 10 neurons (digits 0–9)

---

## 🧪 Training Details

- Loss function: CrossEntropyLoss
- Optimizer: Adam
- Epochs: 5
- Batch size: 64
- Metrics: Accuracy, Classification Report (Precision, Recall, F1-score)
- Validation split: 80/20 (from training data)

---

## 📈 Metrics Tracked

- Training loss per epoch
- Training accuracy per epoch
- Classification report on validation set
- (Optional) Confusion matrix, validation loss, etc.

---

## 📤 Kaggle Submission

Predictions for the test set are saved to `submission.csv` in the required format:

