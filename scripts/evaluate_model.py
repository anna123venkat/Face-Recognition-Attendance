# evaluate_model.py

import os
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Load the trained model and test data
MODEL_PATH = 'app/models/face_recognition_model.pkl'
TEST_DATA_PATH = 'data/dataset/processed/'

def load_model(model_path):
    """Load the trained model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)

def load_test_data(test_data_path):
    """Load test data for evaluation."""
    features = []
    labels = []
    for label_folder in os.listdir(test_data_path):
        label_path = os.path.join(test_data_path, label_folder)
        for image_file in os.listdir(label_path):
            image_path = os.path.join(label_path, image_file)
            image = plt.imread(image_path).ravel()
            features.append(image)
            labels.append(label_folder)
    return np.array(features), np.array(labels)

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on test data."""
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1_score": f1_score(y_test, y_pred, average='weighted'),
    }
    return y_pred, metrics

def plot_confusion_matrix(y_test, y_pred, labels):
    """Plot the confusion matrix."""
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    # Annotate the matrix
    thresh = cm.max() / 2
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'), 
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Loading model and test data...")
    model = load_model(MODEL_PATH)
    X_test, y_test = load_test_data(TEST_DATA_PATH)

    print("Evaluating model...")
    y_pred, metrics = evaluate_model(model, X_test, y_test)
    print("Model Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Plotting confusion matrix...")
    plot_confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
