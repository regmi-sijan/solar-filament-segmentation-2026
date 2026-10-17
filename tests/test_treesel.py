"""Checks for tree-based candidate selection (synthetic data, CPU only)."""
import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import cv2
    from pycocotools import mask as coco
    from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: F401
    from filament.dl import count_matches, encode_instances_fast
    from filament.synthetic import make_synthetic_dataset
    from filament.treesel import (FEATURE_NAMES, NodeScorer, TreeConfig, build_bank, build_tree, node_features,
                                  node_hits, node_mask, run_treesel, score_selection, select_antichain,
                                  select_instances)
except ImportError:  # scikit-learn / torch are optional for the classical pipeline
    cv2 = None


def blob_map():
    """Two dim blobs with brighter cores (so each splits when the threshold rises) and one uniform bright blob."""
    prob = np.zeros((120, 160), np.float32)
    prob[10:40, 10:62] = 0.35
    prob[15:30, 15:35] = 0.7
    prob[15:30, 45:58] = 0.7           # second core: the dim blob splits in two at higher levels
    prob[70:100, 90:140] = 0.9          # uniform: identical at every level
    return prob


def rle(mask):
    return coco.encode(np.asfortranarray(mask.astype(np.uint8)))


@unittest.skipIf(cv2 is None, "scikit-learn is not installed")
class TreeSelectionTests(unittest.TestCase):
    cfg = TreeConfig(levels=(0.2, 0.5, 0.8), min_area=20, disk_erode=0, ring=3)

    def tree(self):
        return build_tree(blob_map(), np.ones((120, 160), np.uint8), self.cfg)

    def test_tree_nests_splits_and_merges_identical_nodes(self):
        nodes = self.tree()["nodes"]
        by_level = {}
        for i, nd in enumerate(nodes):
            by_level.setdefault(nd["level"], []).append(i)
        self.assertEqual(len(by_level[0]), 2)       # dim blob (0.2) and the uniform blob
        self.assertEqual(len(by_level[1]), 2)       # two cores of the dim blob; the uniform blob is not new
        self.assertNotIn(2, by_level)               # at 0.8 only the uniform blob remains and it is identical
        uniform = [i for i in by_level[0] if nodes[i]["area"] == 30 * 50][0]
        self.assertEqual(nodes[uniform]["last_level"], 2)   # remembers that it persists up to 0.8
        for i in by_level[1]:
            self.assertEqual(nodes[nodes[i]["parent"]]["level"], 0)
        for i, nd in enumerate(nodes):
            self.assertLess(nd["parent"], i)        # parents precede children

    def test_features_are_finite_and_named(self):
        tree = self.tree()
        feats = node_features(tree, np.ones((120, 160), np.uint8))
        self.assertEqual(feats.shape, (len(tree["nodes"]), len(FEATURE_NAMES)))
        self.assertTrue(np.isfinite(feats).all())

    def test_select_antichain_is_optimal_against_brute_force(self):
        rng = np.random.default_rng(3)
        for _ in range(60):
            n = int(rng.integers(1, 9))
            parent = np.array([-1 if i == 0 or rng.random() < 0.25 else int(rng.integers(0, i)) for i in range(n)])
            gain = rng.normal(0.0, 1.0, n)
            selectable = rng.random(n) < 0.85

            def related(a, b):
                while a >= 0:
                    if a == b:
                        return True
                    a = parent[a]
                return False

            best = 0.0
            for r in range(n + 1):
                for combo in itertools.combinations([i for i in range(n) if selectable[i]], r):
                    if any(related(a, b) or related(b, a) for a, b in itertools.combinations(combo, 2)):
                        continue
                    best = max(best, sum(gain[i] for i in combo))
            chosen = select_antichain(parent, gain, selectable)
            self.assertAlmostEqual(sum(gain[i] for i in chosen), best, places=9)
            for a, b in itertools.combinations(chosen, 2):
                self.assertFalse(related(a, b) or related(b, a))

    def test_hit_scoring_matches_rle_scoring(self):
        prob, disk = blob_map(), np.ones((120, 160), np.uint8)
        gt_a = np.zeros(prob.shape, bool); gt_a[14:31, 14:36] = True      # roughly the first core
        gt_b = np.zeros(prob.shape, bool); gt_b[70:100, 90:140] = True    # the uniform blob exactly
        gt_c = np.zeros(prob.shape, bool); gt_c[60:64, 5:30] = True       # nothing predicts this
        gt_sets = [("r1", [rle(gt_a), rle(gt_b)]), ("r2", [rle(gt_b), rle(gt_c)])]
        tree = build_tree(prob, disk, self.cfg)
        bank = build_bank("x.jpeg", prob, disk, self.cfg, gt_sets)
        for choice in ([], [0], [0, 1], list(range(len(tree["nodes"])))):
            # restrict to disjoint choices by taking the antichain of the requested nodes
            parent = bank["parent"]
            chosen = [i for i in choice if not any(j in choice and j != i and self._is_ancestor(parent, j, i) for j in choice)]
            rows = score_selection(bank, chosen)
            labels = np.zeros(prob.shape, np.int32)
            for k, i in enumerate(chosen, 1):
                nd = tree["nodes"][i]
                labels[nd["y"]:nd["y"] + nd["h"], nd["x"]:nd["x"] + nd["w"]][node_mask(tree, i)] = k
            pred = encode_instances_fast(labels, list(range(1, len(chosen) + 1)))
            for row, (rid, masks) in zip(rows, gt_sets):
                ref = count_matches(masks, pred)
                self.assertEqual((row["tp"], row["fp"], row["fn"]), (ref["tp"], ref["fp"], ref["fn"]), (choice, rid))
                self.assertAlmostEqual(row["sum_iou"], ref["sum_iou"], places=9)
        self.assertGreater(bank["u"].max(), 0.9)    # the uniform blob matches both records perfectly

    @staticmethod
    def _is_ancestor(parent, a, b):
        while b >= 0:
            if parent[b] == a:
                return True
            b = parent[b]
        return False

    def test_select_instances_returns_disjoint_instances(self):
        bank = build_bank("x.jpeg", blob_map(), np.ones((120, 160), np.uint8), self.cfg,
                          [("r1", [rle(np.pad(np.ones((30, 50), bool), ((70, 20), (90, 20))))])])
        n = len(bank["parent"])
        scorer = NodeScorer("direct", max_iter=5, min_samples_leaf=1).fit(
            bank["features"], bank["u"], bank["h"], np.ones(n))
        labels, accepted = select_instances(blob_map(), np.ones((120, 160), np.uint8), self.cfg, scorer, 0.1)
        self.assertEqual(sorted(accepted), list(range(1, len(accepted) + 1)))
        self.assertEqual(labels.max(), len(accepted))

    def test_end_to_end_on_synthetic_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, out, probs = Path(tmp, "data"), Path(tmp, "out"), Path(tmp, "probs")
            split = make_synthetic_dataset(data, n_train=24, n_test=3, size=128)
            # a stand-in probability map: dark pixels of the (synthetic) image
            for sub, folder in (("train/train_images", "train"), ("test/test_images", "test")):
                for path in Path(data, sub).iterdir():
                    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
                    disk = cv2.GaussianBlur(img, (0, 0), 6)
                    p = np.clip((disk - img) / 60.0, 0, 1)
                    for member in ("a", "b"):
                        target = probs / member
                        target.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(target / f"{path.stem}.png"), np.round(p * 255).astype(np.uint8))
            cfg = TreeConfig(levels=(0.1, 0.25, 0.4, 0.6), min_area=8, disk_erode=2, ring=3)
            metrics = run_treesel(data, split, probs, ["a", "b"], out, cfg=cfg, workers=2, debug=True,
                                  refit={"probs_dir": probs, "members": ["a"]}, log=lambda *_: None)
            self.assertIn(metrics["mode"], ("hurdle", "direct"))
            self.assertTrue(0.0 <= metrics["validation"]["pq"] <= 1.0)
            self.assertTrue(0.0 <= metrics["oracle"]["validation"]["best_pq"] <= 1.0)
            for label in ("held_out_models", "refit_models"):
                self.assertEqual(metrics["csv"][label]["status"], "passed", label)
                self.assertTrue((out / f"submission_{label}.csv").exists())
            self.assertTrue((out / "scorer.pkl").exists())


if __name__ == "__main__":
    unittest.main()
