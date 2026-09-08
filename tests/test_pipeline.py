"""
Sanity and Unit Tests for Animal Disease Models Pipeline
"""
import unittest
from pathlib import Path
import sys
import torch
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import (
    PROJECT_ROOT,
    JUDGE_CLASSES_3,
    AUDIO_MAP,
    SUPPORTED_LANGUAGES,
    JUDGE_IMG_SIZE,
    CP_ANEMIC_SHEET,
    GOAT_DETECTION_RAW
)
from models.judge import MobileNetJudge, AnemiaJudge
from models.finder import EyeFinder


class TestAnimalDiseaseModels(unittest.TestCase):
    def test_01_paths_and_raw_datasets_exist(self):
        """Ensure raw datasets are present."""
        self.assertTrue(CP_ANEMIC_SHEET.exists(), f"Excel sheet missing: {CP_ANEMIC_SHEET}")
        self.assertTrue(GOAT_DETECTION_RAW.exists(), f"Goat dataset missing: {GOAT_DETECTION_RAW}")

    def test_02_audio_triggers_complete(self):
        """Ensure all traffic light classes have audio notes for all Ghanaian languages."""
        for cls_name in JUDGE_CLASSES_3:
            self.assertIn(cls_name, AUDIO_MAP)
            for lang in SUPPORTED_LANGUAGES:
                self.assertIn(lang, AUDIO_MAP[cls_name])
                audio_file = AUDIO_MAP[cls_name][lang]
                self.assertTrue(audio_file.endswith(".mp3"))

    def test_03_judge_model_architecture(self):
        """Test MobileNetV3 Judge forward pass and output shape."""
        model = MobileNetJudge(num_classes=3, pretrained=False)
        dummy_input = torch.randn(2, 3, JUDGE_IMG_SIZE, JUDGE_IMG_SIZE)
        output = model(dummy_input)
        self.assertEqual(output.shape, (2, 3))

    def test_04_judge_inference_wrapper(self):
        """Test AnemiaJudge prediction and probability normalization."""
        judge = AnemiaJudge(model_path=None, num_classes=3, device="cpu")
        dummy_crop = Image.new("RGB", (200, 100), color=(180, 50, 50))
        pred_class, conf, probs = judge.predict(dummy_crop)

        self.assertIn(pred_class, JUDGE_CLASSES_3)
        self.assertTrue(0.0 <= conf <= 1.0)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=4)

    def test_05_finder_crop_padding_and_bounds(self):
        """Test EyeFinder crop function with padding clamping."""
        img = Image.new("RGB", (500, 500), color=(100, 100, 100))
        # Bbox near top-left edge
        bbox = [10.0, 10.0, 60.0, 60.0]
        crop = EyeFinder.crop_eye(img, bbox, padding=0.20)
        self.assertIsInstance(crop, Image.Image)
        self.assertGreater(crop.size[0], 0)
        self.assertGreater(crop.size[1], 0)


if __name__ == "__main__":
    unittest.main()
