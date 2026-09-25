import unittest
import numpy as np
from PIL import Image

import app


class TestAgriAIApp(unittest.TestCase):

    def test_class_names_loaded(self):
        self.assertEqual(len(app.CLASS_NAMES), 15)
        self.assertEqual(len(app.READABLE_CLASS_NAMES), 15)
        self.assertIn("Potato - Early Blight", app.READABLE_CLASS_NAMES)
        self.assertIn("Tomato - Healthy", app.READABLE_CLASS_NAMES)

    def test_readable_label(self):
        self.assertEqual(app.readable_label("Potato___Early_blight"), "Potato - Early Blight")
        self.assertEqual(app.readable_label("Custom_Disease"), "Custom Disease")

    def test_preprocess_shape_and_type(self):
        img = Image.new("RGB", (300, 300), color="green")
        tensor = app.preprocess(img)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        self.assertEqual(tensor.dtype, np.float32)

    def test_predict_none_image(self):
        report, results = app.predict(None)
        self.assertIn("Please upload a plant leaf image", report)
        self.assertEqual(results, {})

    def test_predict_valid_image(self):
        img = Image.new("RGB", (224, 224), color=(100, 150, 200))
        report, results = app.predict(img)
        self.assertIn("# 🌱 AgriAI Analysis", report)
        self.assertEqual(len(results), 3)
        prob_sum = sum(results.values())
        self.assertTrue(0.0 <= prob_sum <= 1.0)


if __name__ == "__main__":
    unittest.main()
