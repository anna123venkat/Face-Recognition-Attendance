import os
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Load the model
MODEL_PATH = 'app/models/face_recognition_model.pkl'

def load_model():
    """Load the trained model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found. Train the model first.")
    return joblib.load(MODEL_PATH)

def test_model_predictions():
    """Test the model's predictions on sample data."""
    model = load_model()

    # Sample test data (Replace with actual test dataset if available)
    test_faces = np.random.rand(5, 2500)  # Simulating 5 test samples of 50x50 images flattened
    test_labels = ['user1', 'user2', 'user1', 'user3', 'user2']  # Expected labels

    # Predict using the model
    predictions = model.predict(test_faces)

    # Evaluate the performance
    print("Predictions:", predictions)
    print("Expected:", test_labels)
    print("Accuracy:", accuracy_score(test_labels, predictions))
    print("Classification Report:\n", classification_report(test_labels, predictions))

def test_model_training():
    """Test if the training process creates the model file correctly."""
    from app.utils.model_utils import train_model  # Assuming train_model is in utils/model_utils.py

    train_model()  # Train the model
    assert os.path.exists(MODEL_PATH), "Model file was not created during training."

if __name__ == "__main__":
    print("Running model tests...\n")

    try:
        print("Testing model predictions...")
        test_model_predictions()
        print("\nTesting model training...")
        test_model_training()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
