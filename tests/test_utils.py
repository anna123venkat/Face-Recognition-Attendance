import unittest
import numpy as np
import pandas as pd
import os
from datetime import datetime
from unittest.mock import patch, MagicMock

# Assuming the utility functions are imported from app.utils
from app.utils import extract_faces, identify_face, train_model, extract_attendance, add_attendance

class TestUtils(unittest.TestCase):

    def setUp(self):
        """Set up resources for the tests."""
        self.sample_image = np.zeros((100, 100, 3), dtype=np.uint8)  # Dummy black image
        self.sample_face_array = np.random.rand(1, 50 * 50)  # Dummy flattened face array
        self.sample_name = "John_123"
        self.sample_csv_path = "data/Attendance/Attendance-test.csv"

        # Create a sample attendance CSV file for testing
        if not os.path.exists("data/Attendance"):
            os.makedirs("data/Attendance")
        pd.DataFrame({
            "Name": ["John"],
            "Roll": [123],
            "Entry Time": ["09:00:00"],
            "Exit Time": ["09:15:00"],
            "Status": ["Present"]
        }).to_csv(self.sample_csv_path, index=False)

    def tearDown(self):
        """Clean up resources after tests."""
        if os.path.exists(self.sample_csv_path):
            os.remove(self.sample_csv_path)

    def test_extract_faces(self):
        """Test the face extraction utility."""
        with patch("cv2.CascadeClassifier.detectMultiScale", return_value=[(10, 10, 50, 50)]) as mock_detect:
            faces = extract_faces(self.sample_image)
            self.assertEqual(len(faces), 1)
            self.assertEqual(faces[0], (10, 10, 50, 50))
            mock_detect.assert_called_once()

    def test_identify_face(self):
        """Test the face identification utility."""
        with patch("joblib.load") as mock_load:
            mock_model = MagicMock()
            mock_model.predict.return_value = ["John_123"]
            mock_load.return_value = mock_model

            result = identify_face(self.sample_face_array)
            self.assertEqual(result[0], "John_123")
            mock_model.predict.assert_called_once_with(self.sample_face_array)

    def test_train_model(self):
        """Test the model training utility."""
        with patch("os.listdir", return_value=["John_123"]):
            with patch("cv2.imread", return_value=self.sample_image):
                with patch("joblib.dump") as mock_dump:
                    train_model()
                    mock_dump.assert_called_once()

    def test_extract_attendance(self):
        """Test the attendance extraction utility."""
        names, rolls, entry, exit, status, l = extract_attendance()
        self.assertEqual(names.iloc[0], "John")
        self.assertEqual(rolls.iloc[0], 123)
        self.assertEqual(entry.iloc[0], "09:00:00")
        self.assertEqual(exit.iloc[0], "09:15:00")
        self.assertEqual(status.iloc[0], "Present")
        self.assertEqual(l, 1)

    def test_add_attendance(self):
        """Test the attendance addition utility."""
        # Mock datetime to return a consistent time
        with patch("datetime.datetime.now", return_value=datetime(2024, 1, 1, 10, 0, 0)):
            add_attendance(self.sample_name)
            df = pd.read_csv(self.sample_csv_path)
            self.assertTrue((df["Name"] == "John").any())
            self.assertTrue((df["Roll"] == 123).any())

if __name__ == "__main__":
    unittest.main()
