"""Checks for the v2 experiment suite (synthetic data, CPU only, a few minutes)."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import cv2
    import torch
    from filament.dl import FilamentUNet, ModelEMA, PostConfig, segment
    from filament.suite import _assign, default_suite, run_suite, summary_table
    from filament.synthetic import make_synthetic_dataset
except ImportError:  # torch is optional for the classical pipeline
    torch = None


def v1_segment(prob, disk, post):
    """The v1 rule, copied verbatim, to check that the new segment() reproduces it."""
    from filament.dl import erode_disk
    binary = ((prob > post.low) & (erode_disk(disk, post.disk_erode) > 0)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    accepted = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < post.min_area or area > post.max_area:
            continue
        if post.high > post.low:
            inside = labels[y:y + h, x:x + w] == label
            if float(prob[y:y + h, x:x + w][inside].max()) < post.high:
                continue
        accepted.append(label)
    return labels, accepted


@unittest.skipIf(torch is None, "torch is not installed")
class SuiteTests(unittest.TestCase):
    def test_segment_matches_v1_when_new_options_are_off(self):
        rng = np.random.default_rng(0)
        disk = np.zeros((256, 256), np.uint8)
        cv2.circle(disk, (128, 128), 120, 1, -1)
        for trial in range(20):
            prob = cv2.GaussianBlur(rng.random((256, 256)).astype(np.float32), (0, 0), 3)
            prob = ((prob - prob.min()) / (prob.max() - prob.min())).astype(np.float16)
            post = PostConfig(low=float(rng.uniform(0.4, 0.7)), min_area=int(rng.integers(1, 80)),
                              disk_erode=int(rng.integers(0, 15)))
            post.high = min(0.95, post.low + float(rng.choice([0.0, 0.1, 0.2])))
            labels_old, kept_old = v1_segment(prob, disk, post)
            labels_new, kept_new = segment(prob, disk, post)
            masks_old = sorted(np.flatnonzero(labels_old == k).tolist() for k in kept_old)
            masks_new = sorted(np.flatnonzero(labels_new == k).tolist() for k in kept_new)
            self.assertEqual(masks_old, masks_new, f"trial {trial}")

    def test_grouping_merges_nearby_fragments_but_not_distant_ones(self):
        prob = np.zeros((100, 200), np.float16)
        prob[40:50, 10:40] = 0.9
        prob[40:50, 46:80] = 0.9     # 6 px gap: same filament
        prob[40:50, 150:190] = 0.9   # 70 px away: separate filament
        disk = np.ones(prob.shape, np.uint8)
        base = PostConfig(min_area=1, disk_erode=0)
        self.assertEqual(len(segment(prob, disk, base)[1]), 3)
        labels, kept = segment(prob, disk, PostConfig(min_area=1, disk_erode=0, group_dist=4))
        self.assertEqual(len(kept), 2)
        self.assertEqual(labels[45, 20], labels[45, 60])
        self.assertTrue(np.all(labels[45, 41:45] == 0), "grouping must not add pixels")

    def test_fill_holes(self):
        prob = np.zeros((60, 60), np.float16)
        prob[10:50, 10:50] = 0.9
        prob[25:35, 25:35] = 0.0
        disk = np.ones(prob.shape, np.uint8)
        labels, kept = segment(prob, disk, PostConfig(min_area=1, disk_erode=0, fill_holes=True))
        self.assertEqual(int((labels == kept[0]).sum()), 1600)

    def test_fast_fill_holes_and_fast_encoding_match_reference_implementations(self):
        from scipy import ndimage
        from filament.core import encode_instances
        from filament.dl import encode_instances_fast, fill_holes
        rng = np.random.default_rng(1)
        for _ in range(10):
            binary = (cv2.GaussianBlur(rng.random((120, 90)).astype(np.float32), (0, 0), 2) > 0.5).astype(np.uint8)
            np.testing.assert_array_equal(fill_holes(binary), ndimage.binary_fill_holes(binary).astype(np.uint8))
            count, labels = cv2.connectedComponents(binary, connectivity=8, ltype=cv2.CV_32S)
            accepted = list(range(1, count))
            self.assertEqual([r["counts"] for r in encode_instances_fast(labels, accepted)],
                             [r["counts"] for r in encode_instances(labels, accepted)])

    def test_resnet50_unet_shapes_and_v1_checkpoint_compatibility(self):
        model = FilamentUNet(pretrained=False, encoder="resnet50").eval()
        with torch.no_grad():
            self.assertEqual(tuple(model(torch.zeros(1, 1, 70, 90)).shape), (1, 1, 70, 90))
        v1_state = FilamentUNet(pretrained=False).state_dict()
        FilamentUNet(pretrained=False, encoder="resnet34").load_state_dict(v1_state)  # strict

    def test_ema_tracks_weights_and_copies_integer_buffers(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3))
        ema = ModelEMA(model, decay=0.5)
        with torch.no_grad():
            model[0].weight.add_(1.0)
        model[1].num_batches_tracked += 7
        ema.update(model)
        state = ema.state_dict()
        self.assertTrue(torch.all(state["0.weight"] != model[0].weight))
        self.assertEqual(int(state["1.num_batches_tracked"]), 7)
        model.load_state_dict(state)

    def test_gpu_assignment_balances_cost(self):
        bins = _assign([{"name": "a", "cost": 1.6}, {"name": "b", "cost": 1.0}, {"name": "c", "cost": 1.0}], 2)
        self.assertEqual(sorted(len(b) for b in bins), [1, 2])

    def test_debug_suite_end_to_end_with_worker_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, out = Path(tmp, "data"), Path(tmp, "out")
            split = make_synthetic_dataset(data, n_train=12, n_test=3, size=128)
            v1 = Path(tmp, "v1_best.pt")
            torch.save(FilamentUNet(pretrained=False).state_dict(), v1)
            metrics = run_suite(data, split, out, default_suite(v1, debug=True), debug=True,
                                probs_root=Path(tmp, "probs"), workers=2, log=lambda *_: None)
            self.assertEqual(len(metrics["candidates"]), 5)  # 3 single models + 2 ensembles
            for cand, result in metrics["candidates"].items():
                self.assertEqual(result["csv"]["status"], "passed", cand)
                self.assertTrue((out / "candidates" / cand / "submission.csv").exists())
                self.assertTrue(0.0 <= result["validation"]["pq"] <= 1.0)
            self.assertIn(metrics["selected_by_calibration"], metrics["candidates"])
            self.assertIn("human_agreement", metrics["diagnostics"])
            self.assertTrue(json.loads((out / "suite_metrics.json").read_text())["debug"])
            self.assertIn("selected", summary_table(metrics))

    def test_debug_refit_trains_on_all_labelled_images_and_writes_a_valid_csv(self):
        from filament.suite import run_refit
        with tempfile.TemporaryDirectory() as tmp:
            data, out = Path(tmp, "data"), Path(tmp, "out")
            split = make_synthetic_dataset(data, n_train=12, n_test=3, size=128)
            v1 = Path(tmp, "v1_best.pt")
            torch.save(FilamentUNet(pretrained=False).state_dict(), v1)
            suite = default_suite(v1, debug=True)
            post = {"low": 0.5, "high": 0.5, "min_area": 10, "max_area": 150000, "disk_erode": 0,
                    "closing_radius": 0, "group_dist": 0, "fill_holes": False}
            metrics = run_refit(data, split, out, suite["models"][:1], post, external=suite["external"],
                                debug=True, probs_root=Path(tmp, "probs"), workers=2, log=lambda *_: None)
            self.assertEqual(metrics["csv"]["status"], "passed")
            self.assertEqual(len(metrics["members"]), 2)
            history = metrics["histories"]["r34_c768_ema"]["history"]
            self.assertTrue(all("tune_pq" not in r for r in history), "no calibration data in a refit")
            self.assertTrue((out / "models" / "r34_c768_ema" / "best.pt").exists())


if __name__ == "__main__":
    unittest.main()
