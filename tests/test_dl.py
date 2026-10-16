"""Checks for the deep-learning extension, using synthetic data only (CPU, about a minute)."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
    from filament.dl import (FilamentUNet, PostConfig, bce_dice_loss, run_experiment, segment,
                             solar_disk)
    from filament.synthetic import make_synthetic_dataset
except ImportError:  # torch is optional for the classical pipeline
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class DeepLearningTests(unittest.TestCase):
    def test_model_handles_sizes_that_are_not_multiples_of_32(self):
        model = FilamentUNet(pretrained=False).eval()
        with torch.no_grad():
            out = model(torch.zeros(1, 1, 100, 70))
        self.assertEqual(tuple(out.shape), (1, 1, 100, 70))

    def test_loss_accepts_soft_targets_and_is_lower_for_a_better_prediction(self):
        target = torch.zeros(1, 1, 8, 8)
        target[..., 2:5, 2:5] = 0.5
        good = torch.where(target > 0, torch.tensor(4.0), torch.tensor(-4.0))
        bad = -good
        self.assertLess(float(bce_dice_loss(good, target)), float(bce_dice_loss(bad, target)))

    def test_hysteresis_and_area_filters(self):
        prob = np.zeros((64, 64), np.float16)
        prob[5:25, 5:25] = 0.9        # strong region, 400 px
        prob[35:55, 35:55] = 0.6      # weak region, 400 px: dropped when high > 0.6
        prob[60:62, 0:3] = 0.95       # tiny region, 6 px: dropped by min_area
        disk = np.ones((64, 64), np.uint8)
        base = PostConfig(low=0.5, high=0.5, min_area=50, disk_erode=0)
        self.assertEqual(len(segment(prob, disk, base)[1]), 2)
        self.assertEqual(len(segment(prob, disk, PostConfig(low=0.5, high=0.8, min_area=50, disk_erode=0))[1]), 1)

    def test_solar_disk_finds_the_bright_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_synthetic_dataset(tmp, n_train=1, n_test=1, size=128)
            import cv2
            image = cv2.imread(str(next(Path(tmp, "train/train_images").iterdir())), cv2.IMREAD_GRAYSCALE)
        disk = solar_disk(image)
        self.assertGreater(disk[64, 64], 0)
        self.assertEqual(disk[2, 2], 0)

    def test_end_to_end_debug_run_writes_a_valid_candidate_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, out = Path(tmp, "data"), Path(tmp, "out")
            split = make_synthetic_dataset(data, n_train=12, n_test=3, size=128)
            metrics = run_experiment(data, split, out, debug=True, log=lambda *_: None)
            self.assertTrue((out / "candidate_submission.csv").exists())
            self.assertEqual(metrics["verification"]["status"], "passed")
            self.assertEqual(metrics["verification"]["test_images"], 3)
            self.assertTrue(0.0 <= metrics["validation"]["pq"] <= 1.0)
            self.assertEqual(json.loads((out / "metrics.json").read_text())["validation"]["images"], 3)


if __name__ == "__main__":
    unittest.main()
