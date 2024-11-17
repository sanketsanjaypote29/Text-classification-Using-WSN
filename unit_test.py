import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

# Importing functions from the Streamlit application
from server import predict_emotions, get_prediction_proba  # Replace 'app' with the actual script name if different

class TestEmotionDetection(unittest.TestCase):
    def setUp(self):
        """Setup for testing."""
        self.sample_text = "I am so happy and excited today!"
        self.mock_prediction = "happy"
        self.mock_proba = np.array([[0.1, 0.2, 0.7]])  # Mock probabilities for three emotions

        # Mocking the pipeline
        self.mock_classes = ["sad", "neutral", "happy"]

    @patch("app.pipe_lr.predict")  # Mock the predict method
    def test_predict_emotions(self, mock_predict):
        """Test the predict_emotions function."""
        mock_predict.return_value = [self.mock_prediction]
        prediction = predict_emotions(self.sample_text)
        mock_predict.assert_called_once_with([self.sample_text])
        self.assertEqual(prediction, self.mock_prediction)

    @patch("app.pipe_lr.predict_proba")  # Mock the predict_proba method
    def test_get_prediction_proba(self, mock_predict_proba):
        """Test the get_prediction_proba function."""
        mock_predict_proba.return_value = self.mock_proba
        probabilities = get_prediction_proba(self.sample_text)
        mock_predict_proba.assert_called_once_with([self.sample_text])
        np.testing.assert_array_almost_equal(probabilities, self.mock_proba)

    def test_prediction_dataframe(self):
        """Test the creation of the probability DataFrame."""
        proba_df = pd.DataFrame(self.mock_proba, columns=self.mock_classes)
        self.assertEqual(proba_df.shape, (1, len(self.mock_classes)))
        self.assertListEqual(list(proba_df.columns), self.mock_classes)

if __name__ == "__main__":
    unittest.main()
