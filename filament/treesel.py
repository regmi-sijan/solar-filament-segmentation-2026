"""Candidate selection on a component tree (replaces a single fixed threshold).

A fixed rule (one threshold, hysteresis, a minimum area) picks one segmentation of a probability
map. Here every connected component at a ladder of thresholds is a *candidate* (a node of a tree:
raising the threshold splits or shrinks a component). A scorer predicts, from the shape and the
probabilities of a candidate, the expected matched IoU ``u`` it would earn against a random
annotator record. Because pooled PQ is ``sum(matched IoU) / (TP + FP/2 + FN/2)``, a prediction
adds to the numerator ``u`` on average and to the denominator ``1/2`` per record, so the best
submission maximises ``sum(u - tau)`` over a set of disjoint candidates, with ``tau`` about
half the achievable PQ. The best disjoint set of a tree (an antichain) is found exactly by dynamic
programming. ``tau`` is tuned on calibration images only.

Only image pixels (through the probability map) and the solar disk feed the scorer; ground truth
is used to train it on calibration images, never at test time.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
from pycocotools import mask as coco

from .dl import erode_disk, fill_holes

LEVELS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80)


@dataclass
class TreeConfig:
    levels: tuple = LEVELS
    min_area: int = 150
    max_area: int = 150000
    disk_erode: int = 10
    fill_holes: bool = True
    ring: int = 10


# --------------------------------------------------------------------------- the tree
def build_tree(prob: np.ndarray, disk: np.ndarray, cfg: TreeConfig) -> dict:
    """Components of ``prob > level`` inside the eroded disk, for every level, linked by inclusion.

    A component identical to its parent (same area) is not a new node; the parent just remembers
    the highest level at which it persists. Returns node arrays plus the label image of each level.
    """
    prob32 = prob.astype(np.float32)
    valid = erode_disk(disk, cfg.disk_erode) > 0
    nodes, node_of, label_images = [], {}, []
    for li, threshold in enumerate(cfg.levels):
        binary = ((prob32 > threshold) & valid).astype(np.uint8)
        if cfg.fill_holes:
            binary = fill_holes(binary)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8, ltype=cv2.CV_32S)
        label_images.append(labels)
        for k in range(1, count):
            x, y, w, h, area = (int(v) for v in stats[k])
            if area < cfg.min_area:
                continue
            parent = -1
            if li > 0:
                inside = labels[y:y + h, x:x + w] == k
                py, px = np.unravel_index(int(np.argmax(inside)), inside.shape)
                parent = node_of.get((li - 1, int(label_images[li - 1][y + py, x + px])), -1)
            if parent >= 0 and nodes[parent]["area"] == area:
                node_of[(li, k)] = parent
                nodes[parent]["last_level"] = li
                continue
            node_of[(li, k)] = len(nodes)
            nodes.append({"level": li, "label": k, "x": x, "y": y, "w": w, "h": h, "area": area,
                          "parent": parent, "last_level": li})
    return {"nodes": nodes, "labels": label_images, "valid": valid, "prob": prob32, "cfg": cfg}


def node_mask(tree: dict, index: int) -> np.ndarray:
    """Boolean mask of one node, cropped to its bounding box."""
    node = tree["nodes"][index]
    crop = tree["labels"][node["level"]][node["y"]:node["y"] + node["h"], node["x"]:node["x"] + node["w"]]
    return crop == node["label"]


FEATURE_NAMES = [
    "t_low", "t_high", "persistence", "area", "log_area", "bbox_long", "bbox_short", "extent",
    "elongation", "major", "minor", "solidity", "compactness",
    "p_mean", "p_max", "p_std", "p_q25", "p_q75", "frac_gt50", "ring_mean", "contrast",
    "area_ratio_parent", "n_children", "n_descendants", "max_child_ratio", "sum_child_ratio", "depth",
    "rho", "disk_activity", "disk_mean_p", "n_nodes_at_half",
]


def node_features(tree: dict, disk: np.ndarray) -> np.ndarray:
    """(N, len(FEATURE_NAMES)) float32 descriptors of every node; image pixels and the disk only."""
    nodes, cfg, prob, valid = tree["nodes"], tree["cfg"], tree["prob"], tree["valid"]
    n = len(nodes)
    if n == 0:
        return np.zeros((0, len(FEATURE_NAMES)), np.float32)
    levels = np.asarray(cfg.levels, np.float32)
    parent = np.array([nd["parent"] for nd in nodes])
    area = np.array([nd["area"] for nd in nodes], np.float64)
    children = [[] for _ in range(n)]
    for i, p in enumerate(parent):
        if p >= 0:
            children[p].append(i)
    n_desc = np.zeros(n)
    depth = np.zeros(n)
    for i in range(n):                      # parents precede children
        if parent[i] >= 0:
            depth[i] = depth[parent[i]] + 1
    for i in range(n - 1, -1, -1):
        n_desc[i] = sum(1 + n_desc[c] for c in children[i])
    moments = cv2.moments(disk.astype(np.uint8), binaryImage=True)
    disk_area = max(moments["m00"], 1.0)
    cx, cy = moments["m10"] / disk_area, moments["m01"] / disk_area
    radius = np.sqrt(disk_area / np.pi)
    valid_area = max(float(valid.sum()), 1.0)
    activity = float(((prob > 0.5) & valid).sum()) / valid_area
    mean_p = float(prob[valid].mean()) if valid.any() else 0.0
    half = float(levels[np.argmin(np.abs(levels - 0.5))])
    at_half = sum(1 for nd in nodes if levels[nd["level"]] <= half <= levels[nd["last_level"]])
    height, width = prob.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cfg.ring + 1, 2 * cfg.ring + 1))
    out = np.zeros((n, len(FEATURE_NAMES)), np.float32)
    for i, nd in enumerate(nodes):
        mask = node_mask(tree, i)
        x, y, w, h = nd["x"], nd["y"], nd["w"], nd["h"]
        values = prob[y:y + h, x:x + w][mask]
        m = cv2.moments(mask.astype(np.uint8), binaryImage=True)
        m00 = max(m["m00"], 1.0)
        mu20, mu02, mu11 = m["mu20"] / m00, m["mu02"] / m00, m["mu11"] / m00
        trace, det = mu20 + mu02, mu20 * mu02 - mu11 ** 2
        disc = np.sqrt(max(trace ** 2 / 4 - det, 0.0))
        l1, l2 = trace / 2 + disc, max(trace / 2 - disc, 1e-3)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = sum(cv2.arcLength(c, True) for c in contours)
        hull = cv2.contourArea(cv2.convexHull(np.vstack(contours))) if contours else nd["area"]
        y0, y1, x0, x1 = max(0, y - cfg.ring), min(height, y + h + cfg.ring), max(0, x - cfg.ring), min(width, x + w + cfg.ring)
        window = np.zeros((y1 - y0, x1 - x0), np.uint8)
        window[y - y0:y - y0 + h, x - x0:x - x0 + w] = mask
        ring = (cv2.dilate(window, kernel) > 0) & (window == 0) & valid[y0:y1, x0:x1]
        ring_mean = float(prob[y0:y1, x0:x1][ring].mean()) if ring.any() else 0.0
        kids = area[children[i]]
        centre = np.hypot(x + w / 2 - cx, y + h / 2 - cy) / radius
        out[i] = [
            levels[nd["level"]], levels[nd["last_level"]], levels[nd["last_level"]] - levels[nd["level"]],
            nd["area"], np.log1p(nd["area"]), max(w, h), min(w, h), nd["area"] / (w * h),
            np.sqrt(l1 / l2), 4 * np.sqrt(l1), 4 * np.sqrt(l2), nd["area"] / max(hull, 1.0),
            4 * np.pi * nd["area"] / max(perimeter, 1.0) ** 2,
            values.mean(), values.max(), values.std(), np.percentile(values, 25), np.percentile(values, 75),
            float((values > 0.5).mean()), ring_mean, float(values.mean()) - ring_mean,
            nd["area"] / area[parent[i]] if parent[i] >= 0 else 1.0, len(kids), n_desc[i],
            kids.max() / nd["area"] if len(kids) else 0.0, kids.sum() / nd["area"] if len(kids) else 0.0, depth[i],
            centre, activity, mean_p, at_half,
        ]
    return out


# --------------------------------------------------------------------------- labels from ground truth
def node_hits(tree: dict, gt_sets: list) -> list:
    """For every record, per node: the ground-truth instances it matches with IoU > 0.5.

    ``gt_sets`` is ``[(record_id, [RLE, ...]), ...]``. Returns, per record, a dict node -> [(gt_index, iou)].
    Candidates of one selection are disjoint, so each ground truth is hit by at most one of them.
    """
    out = []
    for _, masks in gt_sets:
        decoded = [coco.decode(m).astype(bool) for m in masks]
        boxes = [coco.toBbox(m) for m in masks]
        areas = [float(coco.area(m)) for m in masks]
        hits = {}
        for i, nd in enumerate(tree["nodes"]):
            x, y, w, h = nd["x"], nd["y"], nd["w"], nd["h"]
            mask = None
            for g, (bx, by, bw, bh) in enumerate(boxes):
                if bx > x + w or bx + bw < x or by > y + h or by + bh < y:
                    continue
                if mask is None:
                    mask = node_mask(tree, i)
                inter = float(np.count_nonzero(decoded[g][y:y + h, x:x + w] & mask))
                iou = inter / (nd["area"] + areas[g] - inter) if inter else 0.0
                if iou > 0.5:
                    hits.setdefault(i, []).append((g, iou))
        out.append(hits)
    return out


def hit_targets(n_nodes: int, hits: list) -> tuple:
    """Per node: u = mean over records of summed matched IoU, h = share of records with a match."""
    u, h = np.zeros(n_nodes), np.zeros(n_nodes)
    for record in hits:
        for i, found in record.items():
            u[i] += sum(iou for _, iou in found)
            h[i] += 1.0
    r = max(len(hits), 1)
    return u / r, h / r


def score_selection(bank: dict, selected: list) -> list:
    """count_matches-style rows (tp, fp, fn, sum_iou) per record for the selected node indices."""
    rows = []
    chosen = set(selected)
    for rid, n_gt, hits in zip(bank["record_ids"], bank["n_gt"], bank["hits"]):
        tp, sum_iou, matched, with_hit = 0, 0.0, set(), 0
        for i in chosen:
            found = hits.get(i)
            if found:
                with_hit += 1
                for g, iou in found:
                    tp += 1
                    sum_iou += iou
                    matched.add(g)
        rows.append({"file_name": bank["name"], "image_record": rid, "tp": tp, "fp": len(chosen) - with_hit,
                     "fn": n_gt - len(matched), "sum_iou": sum_iou})
    return rows


# --------------------------------------------------------------------------- scorer
class NodeScorer:
    """Expected matched IoU of a candidate. ``hurdle``: P(match) x E[IoU | match]; ``direct``: regress u."""

    def __init__(self, mode: str = "hurdle", max_iter: int = 250, learning_rate: float = 0.05,
                 max_leaf_nodes: int = 15, min_samples_leaf: int = 40, l2: float = 1.0, seed: int = 0):
        self.mode = mode
        self.params = dict(max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
                           min_samples_leaf=min_samples_leaf, l2_regularization=l2, random_state=seed,
                           early_stopping=False)
        self.models = {}

    def _regressor(self):
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(**self.params)

    def fit(self, features: np.ndarray, u: np.ndarray, h: np.ndarray, weight=None) -> "NodeScorer":
        weight = np.ones(len(u)) if weight is None else np.asarray(weight, float)
        if self.mode == "direct":
            self.models = {"u": self._regressor().fit(features, u, sample_weight=weight)}
        else:
            self.models = {"h": self._regressor().fit(features, h, sample_weight=weight)}
            matched = h > 0
            if matched.sum() >= 10:
                self.models["m"] = self._regressor().fit(features[matched], u[matched] / h[matched],
                                                         sample_weight=weight[matched] * h[matched])
            else:
                self.models["m"] = None
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not len(features):
            return np.zeros(0)
        if self.mode == "direct":
            return np.clip(self.models["u"].predict(features), 0.0, 1.0)
        h = np.clip(self.models["h"].predict(features), 0.0, 1.0)
        m = np.full(len(features), 0.65) if self.models["m"] is None else np.clip(self.models["m"].predict(features), 0.5, 1.0)
        return h * m


# --------------------------------------------------------------------------- exact selection
def select_antichain(parent, gain, selectable) -> list:
    """Indices of disjoint nodes maximising sum(gain). Parents must precede children (parent[i] < i)."""
    n = len(parent)
    child_sum = np.zeros(n)
    choose = np.zeros(n, bool)
    for i in range(n - 1, -1, -1):
        if selectable[i] and gain[i] > 0 and gain[i] > child_sum[i]:
            choose[i], best = True, gain[i]
        else:
            best = child_sum[i]
        if parent[i] >= 0:
            child_sum[parent[i]] += best
    selected, blocked = [], np.zeros(n, bool)   # blocked: the node or an ancestor is already selected
    for i in range(n):
        if parent[i] >= 0 and blocked[parent[i]]:
            blocked[i] = True
        elif choose[i]:
            selected.append(i)
            blocked[i] = True
    return selected


# --------------------------------------------------------------------------- banks and end to end
def build_bank(name: str, prob: np.ndarray, disk: np.ndarray, cfg: TreeConfig, gt_sets=None) -> dict:
    """Tree, features and (when ground truth is given) per-record hits for one image. No label images kept."""
    tree = build_tree(prob, disk, cfg)
    nodes = tree["nodes"]
    bank = {"name": name, "features": node_features(tree, disk),
            "parent": np.array([nd["parent"] for nd in nodes], np.int64),
            "area": np.array([nd["area"] for nd in nodes], np.int64),
            "selectable": np.array([nd["area"] <= cfg.max_area for nd in nodes], bool),
            "level": np.array([nd["level"] for nd in nodes], np.int64)}
    if gt_sets is not None:
        hits = node_hits(tree, gt_sets)
        u, h = hit_targets(len(nodes), hits)
        bank.update(record_ids=[rid for rid, _ in gt_sets], n_gt=[len(m) for _, m in gt_sets], hits=hits,
                    u=u, h=h, n_records=len(gt_sets))
    return bank


def select_instances(prob: np.ndarray, disk: np.ndarray, cfg: TreeConfig, scorer: NodeScorer, tau: float):
    """Probability map -> (label image, accepted labels), as dl.segment returns them."""
    tree = build_tree(prob, disk, cfg)
    nodes = tree["nodes"]
    labels = np.zeros(prob.shape, np.int32)
    if not nodes:
        return labels, []
    features = node_features(tree, disk)
    parent = np.array([nd["parent"] for nd in nodes])
    selectable = np.array([nd["area"] <= cfg.max_area for nd in nodes])
    chosen = select_antichain(parent, scorer.predict(features) - tau, selectable)
    for k, i in enumerate(chosen, 1):
        nd = tree["nodes"][i]
        window = labels[nd["y"]:nd["y"] + nd["h"], nd["x"]:nd["x"] + nd["w"]]
        window[node_mask(tree, i)] = k
    return labels, list(range(1, len(chosen) + 1))


def config_dict(cfg: TreeConfig) -> dict:
    out = asdict(cfg)
    out["levels"] = list(out["levels"])
    return out


# --------------------------------------------------------------------------- experiment orchestration
import json
import os
import pickle
import time
from multiprocessing import get_context
from pathlib import Path

from .core import aggregate

_BANK_STATE: dict = {}
_TEST_STATE: dict = {}


def _bank_job(index: int) -> dict:
    item = _BANK_STATE["items"][index]
    prob = _BANK_STATE["load"](item["name"])
    disk = _BANK_STATE["disk"](item["path"])
    return build_bank(item["name"], prob, disk, _BANK_STATE["cfg"], item.get("gt"))


def build_banks(items: list, load, disk_fn, cfg: TreeConfig, workers: int) -> list:
    """One bank per image, built in a fork pool. ``load(name)`` returns the probability map."""
    _BANK_STATE.update(items=items, load=load, disk=disk_fn, cfg=cfg)
    try:
        if workers > 1:
            with get_context("fork").Pool(workers) as pool:
                return pool.map(_bank_job, range(len(items)), chunksize=1)
        return [_bank_job(i) for i in range(len(items))]
    finally:
        _BANK_STATE.clear()


def _stack(banks: list) -> tuple:
    features = np.vstack([b["features"] for b in banks if len(b["features"])])
    keep = [b for b in banks if len(b["features"])]
    u = np.concatenate([b["u"] for b in keep])
    h = np.concatenate([b["h"] for b in keep])
    w = np.concatenate([np.full(len(b["u"]), float(b["n_records"])) for b in keep])
    return features, u, h, w


def fit_scorer(banks: list, mode: str, **kwargs) -> NodeScorer:
    features, u, h, w = _stack(banks)
    return NodeScorer(mode, **kwargs).fit(features, u, h, w)


def crossfit(banks: list, mode: str, folds: int = 5, **kwargs) -> list:
    """Out-of-fold expected-IoU predictions; whole capture dates stay in one fold."""
    keys = [b["name"][:8] for b in banks]
    if len(set(keys)) < 2:          # fewer than two capture dates (tiny test sets): fall back to single images
        keys = [b["name"] for b in banks]
    dates = sorted(set(keys))
    fold_of = {d: i % folds for i, d in enumerate(dates)}
    out = [None] * len(banks)
    for fold in range(min(folds, len(dates))):
        train = [b for b, k in zip(banks, keys) if fold_of[k] != fold]
        scorer = fit_scorer(train, mode, **kwargs)
        for i, k in enumerate(keys):
            if fold_of[k] == fold:
                out[i] = scorer.predict(banks[i]["features"])
    return out


def evaluate_tau(banks: list, uhat: list, tau: float) -> tuple:
    """Pooled PQ of the exact selection at ``tau`` (rows are returned for bootstrapping)."""
    rows = []
    for bank, u in zip(banks, uhat):
        chosen = select_antichain(bank["parent"], u - tau, bank["selectable"]) if len(u) else []
        rows += score_selection(bank, chosen)
    return aggregate(rows)["pq"], rows


def sweep_tau(banks: list, uhat: list, taus) -> dict:
    return {round(float(t), 4): evaluate_tau(banks, uhat, t)[0] for t in taus}


def _test_bank_job(index: int) -> dict:
    from .dl import solar_disk
    st = _TEST_STATE
    path = st["paths"][index]
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return build_bank(Path(path).name, st["load"](Path(path).name), solar_disk(img), st["cfg"])


def _test_mask_job(index: int) -> list:
    from .dl import encode_instances_fast, solar_disk
    st = _TEST_STATE
    path = st["paths"][index]
    chosen = st["chosen"][index]
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    tree = build_tree(st["load"](Path(path).name), solar_disk(img), st["cfg"])
    labels = np.zeros(img.shape, np.int32)
    for k, i in enumerate(chosen, 1):
        nd = tree["nodes"][i]
        labels[nd["y"]:nd["y"] + nd["h"], nd["x"]:nd["x"] + nd["w"]][node_mask(tree, i)] = k
    return [(f"{Path(path).stem}_{n}", rle["counts"].decode("ascii"))
            for n, rle in enumerate(encode_instances_fast(labels, list(range(1, len(chosen) + 1))), 1)]


def _map(job, count: int, workers: int) -> list:
    if workers > 1:
        with get_context("fork").Pool(workers) as pool:
            return pool.map(job, range(count), chunksize=2)
    return [job(i) for i in range(count)]


def write_tree_csv(test_paths: list, load, cfg: TreeConfig, scorer: NodeScorer, tau: float, out_csv, workers: int) -> None:
    """Workers never run scikit-learn (its OpenMP threads deadlock after a fork): workers build banks,
    the parent scores and selects, workers cut the chosen masks."""
    paths = [str(p) for p in test_paths]
    _TEST_STATE.update(paths=paths, load=load, cfg=cfg)
    try:
        banks = _map(_test_bank_job, len(paths), workers)
        _TEST_STATE["chosen"] = [
            select_antichain(b["parent"], scorer.predict(b["features"]) - tau, b["selectable"]) if len(b["parent"]) else []
            for b in banks]
        parts = _map(_test_mask_job, len(paths), workers)
    finally:
        _TEST_STATE.clear()
    with Path(out_csv).open("w") as handle:
        handle.write("filament_id,segmentation_rle\n")
        for part in parts:
            for fid, counts in part:
                handle.write(f"{fid},{counts}\n")


def run_treesel(data_dir, split: dict, probs_dir, members: list, out_dir, baseline_post=None, refit=None,
                cfg=None, workers=None, taus=None, debug: bool = False, log=print) -> dict:
    """Tune on calibration (cross-fitted), score validation once, write test CSVs. Nothing is uploaded.

    ``probs_dir``/``members``: cached probability maps of models trained WITHOUT the calibration and
    validation images, for the calibration, validation and test images. ``refit``: optional
    ``{"probs_dir": ..., "members": [...]}`` of models retrained on all labelled images, used for a
    second test CSV (the scorer is then trained on calibration + validation maps of the held-out models).
    """
    from .core import load_annotations
    from .dl import bootstrap_ci, list_images, load_cases, solar_disk, verify_submission
    from .suite import Evaluator, _load_probs

    cv2.setNumThreads(1)
    data_dir, out_dir, probs_dir = Path(data_dir), Path(out_dir), Path(probs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = cfg or TreeConfig()
    workers = workers or max(1, min(4, os.cpu_count() or 1))
    taus = list(taus) if taus is not None else [round(0.06 + 0.02 * k, 2) for k in range(20)]
    t0 = time.time()
    tune_names, val_names = list(split["calibration"]), list(split["validation"])
    test_paths = list_images(data_dir / "test/test_images")
    if debug:
        tune_names, val_names, test_paths = tune_names[:6], val_names[:4], test_paths[:3]
    records, annotations = load_annotations(data_dir)
    tune_cases = load_cases(data_dir, tune_names, records, annotations, log)
    val_cases = load_cases(data_dir, val_names, records, annotations, log)
    train_dir = data_dir / "train/train_images"

    def loader(dir_, mem):
        return lambda name: _load_probs(Path(dir_), mem, [name])[name]

    def disk_of(path):
        return solar_disk(cv2.imread(path, cv2.IMREAD_GRAYSCALE))

    def items(cases):
        return [{"name": c["name"], "path": str(train_dir / c["name"]), "gt": c["gt"]} for c in cases]

    held_out = loader(probs_dir, members)
    log(f"treesel: building banks for {len(tune_cases)} calibration + {len(val_cases)} validation images")
    banks_cal = build_banks(items(tune_cases), held_out, disk_of, cfg, workers)
    banks_val = build_banks(items(val_cases), held_out, disk_of, cfg, workers)
    nodes_cal = sum(len(b["parent"]) for b in banks_cal)
    log(f"  nodes: calibration {nodes_cal}, validation {sum(len(b['parent']) for b in banks_val)} "
        f"({time.time() - t0:.0f}s)")

    # oracle: the best any scorer could do with this tree (true expected IoU as the score)
    oracle = {}
    for name, banks in (("calibration", banks_cal), ("validation", banks_val)):
        sweep = sweep_tau(banks, [b["u"] for b in banks], taus)
        oracle[name] = {"best_pq": max(sweep.values()), "best_tau": max(sweep, key=sweep.get)}
    log(f"  oracle (true u as the score): {oracle}")

    # choose scorer mode and tau on calibration only, with out-of-fold scores
    tuning = {}
    for mode in ("hurdle", "direct"):
        uhat = crossfit(banks_cal, mode)
        sweep = sweep_tau(banks_cal, uhat, taus)
        tuning[mode] = {"sweep": sweep, "best_tau": max(sweep, key=sweep.get), "best_pq": max(sweep.values())}
        log(f"  {mode}: calibration out-of-fold PQ {tuning[mode]['best_pq']:.4f} at tau {tuning[mode]['best_tau']}")
    mode = max(tuning, key=lambda m: tuning[m]["best_pq"])
    # smooth the tau choice: the centre of the taus within 0.002 PQ of the best
    sweep = tuning[mode]["sweep"]
    good = [t for t, v in sweep.items() if v >= tuning[mode]["best_pq"] - 0.002]
    tau = float(np.median(good))
    log(f"  chosen on calibration: mode={mode} tau={tau:.3f} (within 0.002 of best: {good})")

    # validation, scored once
    scorer = fit_scorer(banks_cal, mode)
    uhat_val = [scorer.predict(b["features"]) for b in banks_val]
    pq_val, rows = evaluate_tau(banks_val, uhat_val, tau)
    summary = aggregate(rows)
    interval, clusters = bootstrap_ci(rows)
    summary.update(pq_bootstrap_95ci=interval, date_clusters=clusters, annotation_sets=len(rows))
    sensitivity = sweep_tau(banks_val, uhat_val, taus)   # reported, never used to choose
    log(f"  VALIDATION pq={summary['pq']:.4f} CI {interval[0]:.4f}-{interval[1]:.4f} "
        f"tp/fp/fn={summary['tp']}/{summary['fp']}/{summary['fn']}")

    baseline = None
    if baseline_post is not None:
        evaluator = Evaluator(val_cases, _load_probs(probs_dir, members, val_names), workers)
        try:
            baseline = aggregate(evaluator.rows(baseline_post))
        finally:
            evaluator.close()
        log(f"  fixed-rule baseline on the same maps: validation pq={baseline['pq']:.4f}")

    # final scorer: every labelled image whose maps come from held-out models
    final_scorer = fit_scorer(banks_cal + banks_val, mode)
    with (out_dir / "scorer.pkl").open("wb") as handle:
        pickle.dump({"scorer": final_scorer, "tau": tau, "cfg": config_dict(cfg), "mode": mode}, handle)
    outputs = {}
    plans = [("held_out_models", probs_dir, members)]
    if refit:
        plans.append(("refit_models", Path(refit["probs_dir"]), refit["members"]))
    for label, dir_, mem in plans:
        csv_path = out_dir / f"submission_{label}.csv"
        write_tree_csv(test_paths, loader(dir_, mem), cfg, final_scorer, tau, csv_path, workers)
        outputs[label] = verify_submission(csv_path, data_dir / "test/test_images")
        log(f"  {label}: {outputs[label]}")

    metrics = {"mode": mode, "tau": tau, "tree_config": config_dict(cfg), "calibration_tuning": tuning,
               "oracle": oracle, "validation": summary, "validation_tau_sensitivity": sensitivity,
               "fixed_rule_baseline_validation": baseline, "csv": outputs, "members": members,
               "nodes_calibration": nodes_cal, "minutes": (time.time() - t0) / 60, "debug": debug}
    (out_dir / "treesel_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    return metrics
