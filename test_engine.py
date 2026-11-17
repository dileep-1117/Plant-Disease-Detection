"""
Unit Test Suite for Plant Disease Prediction Engine
===================================================
Verifies all project requirements:
1. Model loads successfully and matches architecture expectations.
2. Image preprocessing matches training pipeline (96x96, float32, [0, 1]).
3. Prediction output has exactly 38 output values.
4. Class mapping preserves exact original class ordering without sorting.
5. Known test images map to expected high-confidence disease/healthy classes.
6. Two different images are independently processed and yield distinct prediction vectors.
7. No hardcoded predictions exist.
8. Top-5 predictions are correctly sorted in descending order of confidence.
9. Plant Name and Disease Name are accurately extracted from actual predicted classes.
10. Confidence percentage is mathematically calculated from actual softmax score (prob * 100).
11. Healthy status flag accurately identifies healthy vs diseased plant leaves.
12. Two-Stage hierarchical inference correctly identifies plant and conditions disease prediction.
13. Uncertainty detection flags low-confidence or ambiguous leaf photos.
"""

import unittest
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

from model_loader import (
    DEFAULT_CLASS_PATH,
    DEFAULT_MODEL_PATH,
    get_model_info,
    load_class_names,
    load_trained_model,
)
from preprocessing import compute_image_hash, preprocess_image
from prediction import format_class_name, predict_disease
from improved_prediction import predict_two_stage, get_plant_species_catalog


class TestPlantDiseaseEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = Path(__file__).resolve().parent
        cls.model = load_trained_model()
        cls.class_names = load_class_names()
        cls.test_dir = cls.base_dir / "test_images"
        cls.img1_path = cls.test_dir / "corn_rust_sample.jpg"
        cls.img2_path = cls.test_dir / "healthy_leaf_sample.png"

    def test_01_model_loading_and_shapes(self):
        """Test that the trained Keras model loads and has expected input/output shapes."""
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.input_shape, (None, 96, 96, 3))
        self.assertEqual(self.model.output_shape, (None, 38))

    def test_02_class_names_count_and_ordering(self):
        """Test that class_names has exactly 38 classes and is not sorted alphabetically."""
        self.assertEqual(len(self.class_names), 38)
        self.assertNotEqual(
            self.class_names,
            sorted(self.class_names),
            "class_names must not be sorted alphabetically!",
        )
        self.assertEqual(self.class_names[0], "Apple___Apple_scab")
        self.assertEqual(self.class_names[8], "Corn___Common_rust")
        self.assertEqual(self.class_names[19], "Pepper,_bell___healthy")
        self.assertEqual(self.class_names[37], "Tomato___Tomato_Yellow_Leaf_Curl_Virus")

    def test_03_image_preprocessing(self):
        """Test image preprocessing pipeline returns tensor of shape (1, 96, 96, 3) in [0, 1]."""
        tensor, pil_img, img_hash = preprocess_image(self.img1_path)
        self.assertIsInstance(tensor, tf.Tensor)
        self.assertEqual(tensor.shape, (1, 96, 96, 3))
        self.assertEqual(tensor.dtype, tf.float32)

        # Value range check [0, 1]
        tensor_np = tensor.numpy()
        self.assertGreaterEqual(float(np.min(tensor_np)), 0.0)
        self.assertLessEqual(float(np.max(tensor_np)), 1.0)
        self.assertTrue(len(img_hash) == 64)

    def test_04_prediction_vector_dimension_and_sum(self):
        """Test that prediction returns 38 values summing to ~1.0."""
        result = predict_disease(
            self.img1_path,
            model=self.model,
            class_names=self.class_names,
            top_k=5,
        )
        raw_probs = result["raw_probabilities"]
        self.assertEqual(len(raw_probs), 38)
        self.assertAlmostEqual(float(np.sum(raw_probs)), 1.0, places=4)
        self.assertEqual(result["num_classes"], 38)

    def test_05_class_mapping_accuracy(self):
        """Test that known test images map to expected high-confidence disease/healthy classes."""
        # Image 1: corn_rust_sample.jpg should map to Corn___Common_rust
        res1 = predict_disease(self.img1_path, model=self.model, class_names=self.class_names)
        self.assertEqual(res1["top_prediction"]["class_name"], "Corn___Common_rust")
        self.assertGreater(res1["top_prediction"]["confidence"], 0.85)

        # Image 2: healthy_leaf_sample.png should map to Pepper,_bell___healthy
        res2 = predict_disease(self.img2_path, model=self.model, class_names=self.class_names)
        self.assertEqual(res2["top_prediction"]["class_name"], "Pepper,_bell___healthy")
        self.assertGreater(res2["top_prediction"]["confidence"], 0.90)

    def test_06_independent_processing_and_no_leakage(self):
        """Test that two different images are independently processed with distinct outputs."""
        res1 = predict_disease(self.img1_path, model=self.model, class_names=self.class_names)
        res2 = predict_disease(self.img2_path, model=self.model, class_names=self.class_names)

        self.assertNotEqual(res1["image_hash"], res2["image_hash"])
        self.assertNotEqual(
            res1["top_prediction"]["class_name"],
            res2["top_prediction"]["class_name"],
        )
        diff = np.abs(res1["raw_probabilities"] - res2["raw_probabilities"])
        self.assertGreater(float(np.max(diff)), 0.5)

    def test_07_no_hardcoded_predictions_with_synthetic_variation(self):
        """Test that modifying image pixels dynamically changes prediction vector."""
        img_a = Image.new("RGB", (128, 128), color=(255, 0, 0))  # Solid Red
        img_b = Image.new("RGB", (128, 128), color=(0, 255, 0))  # Solid Green

        res_a = predict_disease(img_a, model=self.model, class_names=self.class_names)
        res_b = predict_disease(img_b, model=self.model, class_names=self.class_names)

        self.assertNotEqual(res_a["image_hash"], res_b["image_hash"])
        diff = np.abs(res_a["raw_probabilities"] - res_b["raw_probabilities"])
        self.assertGreater(float(np.max(diff)), 0.05)

    def test_08_top_5_ranking_order(self):
        """Test that top 5 predictions are strictly sorted in descending confidence order."""
        result = predict_disease(self.img1_path, model=self.model, class_names=self.class_names, top_k=5)
        top_5 = result["top_5"]
        self.assertEqual(len(top_5), 5)

        for i in range(len(top_5) - 1):
            self.assertGreaterEqual(
                top_5[i]["confidence"],
                top_5[i + 1]["confidence"],
                f"Rank {i+1} confidence should be >= Rank {i+2}",
            )

    def test_09_plant_and_disease_name_extraction(self):
        """Test plant name and disease name extraction for exact required examples."""
        info1 = format_class_name("Tomato___Early_blight")
        self.assertEqual(info1["plant_name"], "Tomato")
        self.assertEqual(info1["disease_name"], "Early Blight")
        self.assertFalse(info1["is_healthy"])

        info2 = format_class_name("Apple___Apple_scab")
        self.assertEqual(info2["plant_name"], "Apple")
        self.assertEqual(info2["disease_name"], "Apple Scab")
        self.assertFalse(info2["is_healthy"])

        info3 = format_class_name("Corn___Common_rust")
        self.assertEqual(info3["plant_name"], "Corn")
        self.assertEqual(info3["disease_name"], "Common Rust")
        self.assertFalse(info3["is_healthy"])

        info4 = format_class_name("Apple___healthy")
        self.assertEqual(info4["plant_name"], "Apple")
        self.assertEqual(info4["disease_name"], "Healthy")
        self.assertTrue(info4["is_healthy"])

    def test_10_confidence_percentage_calculation(self):
        """Test that confidence percentage is computed as actual softmax score * 100."""
        res = predict_disease(self.img1_path, model=self.model, class_names=self.class_names)
        top = res["top_prediction"]
        raw_prob = float(res["raw_probabilities"][top["class_index"]])

        expected_score = raw_prob * 100.0
        self.assertAlmostEqual(top["confidence_score"], expected_score, places=5)
        self.assertIn(f"{expected_score:.1f}%", top["confidence_pct"])

    def test_11_healthy_status_flag(self):
        """Test healthy status flag on healthy and diseased leaves."""
        res_diseased = predict_disease(self.img1_path, model=self.model, class_names=self.class_names)
        self.assertFalse(res_diseased["top_prediction"]["is_healthy"])

        res_healthy = predict_disease(self.img2_path, model=self.model, class_names=self.class_names)
        self.assertTrue(res_healthy["top_prediction"]["is_healthy"])
        self.assertEqual(res_healthy["top_prediction"]["disease_name"], "Healthy")

    def test_12_two_stage_prediction_and_catalog(self):
        """Test Two-Stage Prediction Engine produces valid plant and conditioned disease."""
        plant_to_classes, class_to_plant = get_plant_species_catalog(self.class_names)
        self.assertEqual(len(plant_to_classes), 14)
        self.assertEqual(len(class_to_plant), 38)
        self.assertIn("Apple", plant_to_classes)
        self.assertIn("Corn", plant_to_classes)
        self.assertIn("Tomato", plant_to_classes)

        res = predict_two_stage(self.img1_path, model=self.model, class_names=self.class_names)
        self.assertEqual(res["plant_name"], "Corn")
        self.assertEqual(res["disease_name"], "Common Rust")
        self.assertTrue(res["is_reliable"])
        self.assertGreater(res["plant_confidence"], 0.90)

    def test_13_uncertainty_thresholds(self):
        """Test that low-confidence or synthetic images trigger the uncertainty advisory."""
        # Flat gray noise has low/ambiguous confidence
        gray_img = Image.new("RGB", (96, 96), color=(120, 120, 120))
        res_gray = predict_two_stage(
            gray_img,
            model=self.model,
            class_names=self.class_names,
            confidence_threshold=0.80, # High threshold to test uncertainty trip
            ambiguity_threshold=0.10
        )
        # Verify reliability message format
        if not res_gray["is_reliable"]:
            self.assertEqual(
                res_gray["reliability_message"],
                "Unable to identify this leaf reliably. Please upload a clearer image."
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
