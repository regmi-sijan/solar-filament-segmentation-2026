"""Experiment suite v2: several U-Nets trained in parallel, ensembled, tuned and compared.

Protocol (same partitions as v1, no leakage):
* every model trains on ``training`` minus ``calibration``;
* checkpoint choice, post-processing and the choice between candidates use ``calibration`` only;
* every candidate is then scored once on ``validation`` and gets its own test CSV.

One worker process per GPU trains (and later runs inference for) its share of the models, so a
Kaggle "GPU T4 x2" session trains two models at once. Probability maps are cached as 8-bit PNGs,
which makes ensembling and post-processing searches cheap and GPU-free.

Nothing is submitted to Kaggle by this module.

    python -m filament.suite train <job.json>     # worker entry points (used internally)
    python -m filament.suite infer <job.json>
"""
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, replace
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as coco

from .core import aggregate, load_annotations
from .dl import (PostConfig, TrainConfig, bootstrap_ci, count_matches, encode_instances_fast, list_images, load_cases,
                 segment, solar_disk, verify_submission)

CODE_ROOT = Path(__file__).resolve().parents[1]

# Post-processing search space; ``margin`` sets high = min(0.95, low + margin).
SEARCH_SPACE = [
    ("low", [0.3, 0.4, 0.5, 0.6, 0.7]),
    ("margin", [0.0, 0.15, 0.3]),
    ("min_area", [100, 300, 600, 1000, 1600]),
    ("group_dist", [0, 4, 8, 16, 32]),
    ("fill_holes", [False, True]),
    ("disk_erode", [0, 10, 20]),
]


# --------------------------------------------------------------------------- suite definition
def default_suite(v1_weights=None, debug: bool = False) -> dict:
    """Two new models (one per T4), plus the v1 checkpoint if it is attached as an input."""
    base = TrainConfig(crop=768, epochs=32, steps_per_epoch=150, batch_size=8, workers=2,
                       ema_decay=0.998, eval_every=4)
    models = [
        {"name": "r34_c768_ema", "cfg": asdict(replace(base, encoder="resnet34", seed=2027)), "cost": 1.0},
        {"name": "r50_c768_ema", "cfg": asdict(replace(base, encoder="resnet50", seed=2028)), "cost": 1.6},
    ]
    if debug:
        for spec in models:
            spec["cfg"].update(crop=256, batch_size=2, epochs=2, steps_per_epoch=3, workers=0,
                               eval_every=1, eval_images=2, pretrained=False)
    external = []
    if v1_weights:
        external.append({"name": "v1_r34_c512", "encoder": "resnet34", "weights": str(v1_weights), "cost": 1.0})
    names = [m["name"] for m in models]
    ensembles = [names]
    if external:
        ensembles.append(names + [external[0]["name"]])
    return {"models": models, "external": external, "ensembles": ensembles}


def _candidates(suite: dict) -> dict:
    """Candidate name -> member model names: every single model, then every ensemble."""
    out = {m["name"]: [m["name"]] for m in suite["models"] + suite["external"]}
    for members in suite["ensembles"]:
        out["ens_" + "+".join(members)] = list(members)
    return out


# --------------------------------------------------------------------------- worker processes
def _log(*parts) -> None:
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def _device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _train_worker(job: dict) -> None:
    import torch
    from .dl import FilamentUNet, seed_everything, train_model

    device = _device()
    records, annotations = load_annotations(job["data"])
    for spec in job["models"]:
        cfg = TrainConfig(**spec["cfg"])
        out_dir = Path(job["out"]) / spec["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        _log(f"[{spec['name']}] device={device} gpu={torch.cuda.get_device_name(0) if device.type == 'cuda' else '-'}")
        seed_everything(cfg.seed)
        train_cases = load_cases(job["data"], job["train_names"], records, annotations, _log)
        for case in train_cases:  # training reads only image, target, mean/std and positives
            case.pop("disk", None)
            case.pop("gt", None)
        tune_cases = load_cases(job["data"], job["tune_names"][: cfg.eval_images], records, annotations, _log)
        model = FilamentUNet(cfg.pretrained, cfg.encoder).to(device)
        start = time.time()
        history = train_model(model, train_cases, tune_cases, cfg, device, out_dir, _log)
        (out_dir / "history.json").write_text(json.dumps(
            {"cfg": asdict(cfg), "history": history, "minutes": (time.time() - start) / 60}, indent=2))
        del train_cases, tune_cases, model
        torch.cuda.empty_cache()
        _log(f"[{spec['name']}] training finished")


def _infer_worker(job: dict) -> None:
    import torch
    from .dl import FilamentUNet, predict_proba

    device = _device()
    cfg = TrainConfig(tta=job.get("tta", True))
    for spec in job["models"]:
        model = FilamentUNet(False, spec["encoder"]).to(device)
        model.load_state_dict(torch.load(spec["weights"], map_location=device))
        out_dir = Path(job["probs"]) / spec["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        start, items = time.time(), sorted(job["images"].items())
        for index, (stem, path) in enumerate(items, 1):
            target = out_dir / f"{stem}.png"
            if target.exists():
                continue
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            prob = predict_proba(model, img, device, cfg).astype(np.float32)
            cv2.imwrite(str(target), np.round(prob * 255).astype(np.uint8))
            if index % 50 == 0 or index == len(items):
                _log(f"[{spec['name']}] inference {index}/{len(items)}")
        _log(f"[{spec['name']}] inference finished in {(time.time() - start) / 60:.1f} min")
        del model
        torch.cuda.empty_cache()


def _assign(items: list, slots: int) -> list:
    """Greedy longest-first assignment of weighted jobs to GPU slots."""
    bins = [[] for _ in range(slots)]
    loads = [0.0] * slots
    for item in sorted(items, key=lambda i: -i.get("cost", 1.0)):
        k = int(np.argmin(loads))
        bins[k].append(item)
        loads[k] += item.get("cost", 1.0)
    return bins


def _gpu_slots() -> list:
    import torch
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return list(range(count)) or [None]


def _tail(path: Path, lines: int = 1) -> str:
    try:
        text = path.read_text(errors="replace").strip().splitlines()
    except FileNotFoundError:
        return "(no output yet)"
    return "\n".join(text[-lines:]) if text else "(no output yet)"


def _launch(kind: str, job_items: list, common: dict, work: Path) -> list:
    """Start one worker per GPU; returns (process, log path, file handle) triples."""
    slots = _gpu_slots()
    procs = []
    for slot, (gpu, items) in enumerate(zip(slots, _assign(job_items, len(slots)))):
        if not items:
            continue
        job_path = work / f"{kind}_job{slot}.json"
        job_path.write_text(json.dumps({**common, "models": items}))
        log_path = work / f"{kind}_worker{slot}.log"
        env = dict(os.environ, PYTHONUNBUFFERED="1",
                   CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu),
                   PYTHONPATH=os.pathsep.join(filter(None, [str(CODE_ROOT), os.environ.get("PYTHONPATH")])))
        handle = log_path.open("w")
        proc = subprocess.Popen([sys.executable, "-u", "-m", "filament.suite", kind, str(job_path)],
                                stdout=handle, stderr=subprocess.STDOUT, env=env)
        procs.append((proc, log_path, handle))
        _log(f"started {kind} worker {slot} on gpu {gpu}: {[i['name'] for i in items]}")
    return procs


def _wait(procs: list, every: float = 300.0) -> None:
    last = time.time()
    while any(p.poll() is None for p, _, _ in procs):
        time.sleep(10)
        if time.time() - last >= every:
            last = time.time()
            for _, log_path, _ in procs:
                _log(f"{log_path.name}: {_tail(log_path)}")
    for proc, log_path, handle in procs:
        handle.close()
        if proc.returncode != 0:
            raise RuntimeError(f"{log_path.name} exited with {proc.returncode}:\n{_tail(log_path, 40)}")


# --------------------------------------------------------------------------- parallel evaluation
_STATE: dict = {}


def _eval_image(args):
    index, post = args
    case = _STATE["cases"][index]
    labels, accepted = segment(_STATE["probs"][case["name"]], case["disk"], PostConfig(**post))
    pred = encode_instances_fast(labels, accepted)
    return [{"file_name": case["name"], "image_record": rid, **count_matches(masks, pred)}
            for rid, masks in case["gt"]]


class Evaluator:
    """Scores PostConfigs on a fixed set of cases and probability maps, in parallel, with a cache."""

    def __init__(self, cases: list, probs: dict, workers: int):
        _STATE.update(cases=cases, probs=probs)
        self.n, self.cache = len(cases), {}
        self.pool = get_context("fork").Pool(workers) if workers > 1 else None

    def rows(self, post: PostConfig) -> list:
        key = tuple(sorted(asdict(post).items()))
        if key not in self.cache:
            args = [(i, asdict(post)) for i in range(self.n)]
            parts = self.pool.map(_eval_image, args, chunksize=2) if self.pool else map(_eval_image, args)
            self.cache[key] = [row for part in parts for row in part]
        return self.cache[key]

    def pq(self, post: PostConfig) -> float:
        return aggregate(self.rows(post))["pq"]

    def close(self) -> None:
        if self.pool:
            self.pool.close()
            self.pool.join()
        _STATE.clear()


def _make_post(params: dict) -> PostConfig:
    return PostConfig(low=params["low"], high=min(0.95, params["low"] + params["margin"]),
                      min_area=params["min_area"], group_dist=params["group_dist"],
                      fill_holes=params["fill_holes"], disk_erode=params["disk_erode"])


def tune_post_v2(evaluator: Evaluator, passes: int = 2, log=_log):
    """Coordinate search over SEARCH_SPACE (calibration images only); returns (PostConfig, pq)."""
    params = {"low": 0.5, "margin": 0.0, "min_area": 300, "group_dist": 0, "fill_holes": False, "disk_erode": 10}
    best = evaluator.pq(_make_post(params))
    log(f"  start pq={best:.4f} {params}")
    for round_index in range(passes):
        improved = False
        for key, values in SEARCH_SPACE:
            for value in values:
                if value == params[key]:
                    continue
                trial = dict(params, **{key: value})
                score = evaluator.pq(_make_post(trial))
                if score > best + 1e-9:
                    params, best, improved = trial, score, True
        log(f"  pass {round_index + 1}: pq={best:.4f} {params}")
        if not improved:
            break
    return _make_post(params), best


# --------------------------------------------------------------------------- probability maps
def _load_probs(probs_dir: Path, members: list, names: list) -> dict:
    out = {}
    for name in names:
        total = None
        for member in members:
            path = probs_dir / member / f"{Path(name).stem}.png"
            arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if arr is None:
                raise FileNotFoundError(path)
            total = arr.astype(np.float32) if total is None else total + arr
        out[name] = (total / (255.0 * len(members))).astype(np.float16)
    return out


def _csv_image(args):
    path, members, probs_dir, post = args
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    prob = _load_probs(Path(probs_dir), members, [Path(path).name])[Path(path).name]
    labels, accepted = segment(prob, solar_disk(img), PostConfig(**post))
    return [(f"{Path(path).stem}_{n}", rle["counts"].decode("ascii"))
            for n, rle in enumerate(encode_instances_fast(labels, accepted), 1)]


def write_candidate_csv(test_paths: list, members: list, probs_dir: Path, post: PostConfig, out_csv: Path,
                        workers: int) -> None:
    args = [(str(p), members, str(probs_dir), asdict(post)) for p in test_paths]
    if workers > 1:
        with get_context("fork").Pool(workers) as pool:
            parts = pool.map(_csv_image, args, chunksize=2)
    else:
        parts = list(map(_csv_image, args))
    with out_csv.open("w") as handle:
        handle.write("filament_id,segmentation_rle\n")
        for part in parts:
            for fid, counts in part:
                handle.write(f"{fid},{counts}\n")


# --------------------------------------------------------------------------- diagnostics
def diagnostics(names: list, records: dict, annotations: dict, tune_cases: list, val_cases: list,
                workers: int, log=_log) -> dict:
    """What limits the score: annotator agreement, multi-part filaments, instance formation."""
    out = {}
    parts = [len(a["segmentation"]) if isinstance(a["segmentation"], list) else 1
             for name in names for record in records[name] for a in annotations[record["id"]]]
    out["annotations"] = len(parts)
    out["share_multi_polygon"] = float(np.mean([p > 1 for p in parts])) if parts else 0.0
    components = []
    for case in tune_cases + val_cases:
        for _, masks in case["gt"]:
            for rle in masks:
                x, y, w, h = [int(v) for v in coco.toBbox(rle)]
                crop = coco.decode(rle)[y:y + h + 1, x:x + w + 1]
                components.append(cv2.connectedComponents(crop, connectivity=8)[0] - 1)
    out["gt_instances_scored"] = len(components)
    out["share_gt_instances_multi_part"] = float(np.mean([c > 1 for c in components])) if components else 0.0
    pairs, multi = [], 0
    for case in tune_cases + val_cases:
        sets = case["gt"]
        multi += len(sets) > 1
        for i, (_, reference) in enumerate(sets):
            for j, (_, other) in enumerate(sets):
                if i != j:
                    pairs.append(count_matches(reference, other))
    out["images_with_2plus_annotators"] = multi
    out["human_agreement"] = aggregate(pairs) if pairs else None
    probs = {c["name"]: (c["target"].astype(np.float32) / 255.0).astype(np.float16) for c in tune_cases}
    evaluator = Evaluator(tune_cases, probs, workers)
    try:
        for group in (0, 8, 16, 32):
            post = PostConfig(low=0.49, high=0.49, min_area=1, disk_erode=0, group_dist=group)
            out[f"consensus_as_prediction_pq_group{group}"] = evaluator.pq(post)
    finally:
        evaluator.close()
    log(f"diagnostics: {json.dumps(out)}")
    return out


# --------------------------------------------------------------------------- orchestration
def run_suite(data_dir, split: dict, out_dir, suite: dict, debug: bool = False, baseline_pq: float = 0.1796,
              v1_val_pq: float = 0.4111, probs_root=None, workers=None, log=_log) -> dict:
    """Train, infer, ensemble, tune on calibration, score validation, write one CSV per candidate."""
    cv2.setNumThreads(1)  # keep OpenCV's thread pool idle in this process: it forks evaluation workers
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "work"
    work.mkdir(exist_ok=True)
    probs_dir = Path(probs_root or "/tmp/filament_probs") / out_dir.name
    workers = workers or max(1, min(4, os.cpu_count() or 1))
    timings, t0 = {}, time.time()

    calibration = set(split["calibration"])
    train_names = [n for n in split["training"] if n not in calibration]
    tune_names, val_names = list(split["calibration"]), list(split["validation"])
    test_paths = list_images(data_dir / "test/test_images")
    if debug:
        train_names, tune_names, val_names, test_paths = train_names[:6], tune_names[:4], val_names[:4], test_paths[:4]
    log(f"suite: train={len(train_names)} calibration={len(tune_names)} validation={len(val_names)} "
        f"test={len(test_paths)} gpus={_gpu_slots()} cpu_workers={workers}")

    # 1. training, one worker per GPU; diagnostics and case loading overlap with it on the CPU
    procs = _launch("train", suite["models"],
                    {"data": str(data_dir), "out": str(out_dir / "models"),
                     "train_names": train_names, "tune_names": tune_names}, work)
    records, annotations = load_annotations(data_dir)
    tune_cases = load_cases(data_dir, tune_names, records, annotations, log)
    val_cases = load_cases(data_dir, val_names, records, annotations, log)
    diag = diagnostics(train_names + tune_names + val_names, records, annotations, tune_cases, val_cases,
                       workers, log)
    _wait(procs)
    timings["train_min"] = (time.time() - t0) / 60
    histories = {m["name"]: json.loads((out_dir / "models" / m["name"] / "history.json").read_text())
                 for m in suite["models"]}

    # 2. inference: cached probability maps for calibration, validation and test images
    t1 = time.time()
    images = {Path(n).stem: str(data_dir / "train/train_images" / n) for n in tune_names + val_names}
    images.update({p.stem: str(p) for p in test_paths})
    infer_models = [{"name": m["name"], "encoder": m["cfg"]["encoder"], "cost": m.get("cost", 1.0),
                     "weights": str(out_dir / "models" / m["name"] / "best.pt")} for m in suite["models"]]
    infer_models += suite["external"]
    _wait(_launch("infer", infer_models, {"probs": str(probs_dir), "images": images, "tta": True}, work))
    timings["infer_min"] = (time.time() - t1) / 60

    # 3. candidates: tune on calibration, choose on calibration, score every candidate on validation
    t2 = time.time()
    results = {}
    for cand, members in _candidates(suite).items():
        log(f"candidate {cand}")
        evaluator = Evaluator(tune_cases, _load_probs(probs_dir, members, tune_names), workers)
        try:
            post, tune_pq = tune_post_v2(evaluator, passes=1 if debug else 2, log=log)
        finally:
            evaluator.close()
        evaluator = Evaluator(val_cases, _load_probs(probs_dir, members, val_names), workers)
        try:
            rows = evaluator.rows(post)
            baseline_rows = evaluator.rows(replace(post, group_dist=0, fill_holes=False))
        finally:
            evaluator.close()
        summary = aggregate(rows)
        interval, clusters = bootstrap_ci(rows)
        summary.update(pq_bootstrap_95ci=interval, date_clusters=clusters, annotation_sets=len(rows))
        results[cand] = {"members": members, "post": asdict(post), "calibration_pq": tune_pq,
                         "validation": summary,
                         "validation_pq_without_grouping": aggregate(baseline_rows)["pq"]}
        log(f"  calibration pq={tune_pq:.4f}  validation pq={summary['pq']:.4f} "
            f"CI {interval[0]:.4f}-{interval[1]:.4f}  tp/fp/fn={summary['tp']}/{summary['fp']}/{summary['fn']}")
    timings["tune_min"] = (time.time() - t2) / 60

    # 4. one candidate CSV per candidate (never uploaded by this code)
    t3 = time.time()
    selected = max(results, key=lambda c: (results[c]["calibration_pq"], -len(results[c]["members"])))
    for cand, result in results.items():
        cand_dir = out_dir / "candidates" / cand
        cand_dir.mkdir(parents=True, exist_ok=True)
        csv_path = cand_dir / "submission.csv"
        write_candidate_csv(test_paths, result["members"], probs_dir, PostConfig(**result["post"]), csv_path, workers)
        result["csv"] = verify_submission(csv_path, data_dir / "test/test_images")
        if debug:
            result["csv"]["note"] = "debug run: only a few test images"
        log(f"{cand}: {result['csv']}")
    (out_dir / "SELECTED_BY_CALIBRATION.txt").write_text(
        f"{selected}\n{results[selected]['csv']['sha256']}\n")
    timings["csv_min"] = (time.time() - t3) / 60
    timings["total_min"] = (time.time() - t0) / 60

    metrics = {"selected_by_calibration": selected, "candidates": results, "diagnostics": diag,
               "histories": histories, "suite": suite, "timings": timings,
               "reference": {"classical_baseline_validation_pq": baseline_pq, "v1_validation_pq": v1_val_pq},
               "debug": debug}
    (out_dir / "suite_metrics.json").write_text(json.dumps(metrics, indent=2))
    log(f"selected by calibration: {selected}; timings {timings}")
    return metrics


def run_refit(data_dir, split: dict, out_dir, models: list, post: dict, external=None, debug: bool = False,
              probs_root=None, workers=None, log=_log) -> dict:
    """Retrain ``models`` on every labelled image (training + validation) and write one test CSV.

    Use this only after a suite run has chosen the configuration and ``post`` on calibration data.
    There is no held-out data left, so the checkpoint is the final (EMA) weights and the reported
    estimate of quality remains the suite's validation score. Nothing is submitted to Kaggle.
    """
    cv2.setNumThreads(1)
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "work"
    work.mkdir(exist_ok=True)
    probs_dir = Path(probs_root or "/tmp/filament_probs") / out_dir.name
    workers = workers or max(1, min(4, os.cpu_count() or 1))
    external = list(external or [])
    train_names = list(split["training"]) + list(split["validation"])
    test_paths = list_images(data_dir / "test/test_images")
    if debug:
        train_names, test_paths = train_names[:6], test_paths[:4]
    log(f"refit: train={len(train_names)} test={len(test_paths)} models={[m['name'] for m in models]} "
        f"external={[e['name'] for e in external]} post={post}")
    t0 = time.time()
    _wait(_launch("train", models, {"data": str(data_dir), "out": str(out_dir / "models"),
                                    "train_names": train_names, "tune_names": []}, work))
    infer_models = [{"name": m["name"], "encoder": m["cfg"]["encoder"], "cost": m.get("cost", 1.0),
                     "weights": str(out_dir / "models" / m["name"] / "last.pt")} for m in models] + external
    _wait(_launch("infer", infer_models, {"probs": str(probs_dir), "tta": True,
                                          "images": {p.stem: str(p) for p in test_paths}}, work))
    members = [m["name"] for m in infer_models]
    csv_path = out_dir / "submission.csv"
    write_candidate_csv(test_paths, members, probs_dir, PostConfig(**post), csv_path, workers)
    verification = verify_submission(csv_path, data_dir / "test/test_images")
    metrics = {"members": members, "post": post, "train_images": len(train_names), "csv": verification,
               "histories": {m["name"]: json.loads((out_dir / "models" / m["name"] / "history.json").read_text())
                             for m in models},
               "minutes": (time.time() - t0) / 60, "debug": debug}
    (out_dir / "refit_metrics.json").write_text(json.dumps(metrics, indent=2))
    log(f"refit done: {verification}")
    return metrics


def compare_submissions(reference_csv, other_csv, test_dir) -> dict:
    """Agreement between two submission files on the test images (no labels involved).

    ``matched_pq`` scores ``other`` against ``reference`` with the competition's strict IoU > 0.5
    matching; ``mean_pixel_dice`` compares the union masks image by image. A sanity check that a
    retrained model behaves like the validated one, not a measure of accuracy.
    """
    import csv as _csv
    from collections import defaultdict

    paths = {p.stem: p for p in list_images(test_dir)}
    shapes = {}

    def read(path):
        out = defaultdict(list)
        with Path(path).open() as handle:
            for row in _csv.DictReader(handle):
                stem = row["filament_id"].rsplit("_", 1)[0]
                if stem not in shapes:
                    shapes[stem] = cv2.imread(str(paths[stem]), cv2.IMREAD_GRAYSCALE).shape
                out[stem].append({"size": list(shapes[stem]), "counts": row["segmentation_rle"].encode("ascii")})
        return out

    ref, other = read(reference_csv), read(other_csv)
    rows, dices = [], []
    for stem in sorted(set(ref) | set(other)):
        a, b = ref.get(stem, []), other.get(stem, [])
        rows.append(count_matches(a, b))
        area_a = float(coco.area(coco.merge(a))) if a else 0.0
        area_b = float(coco.area(coco.merge(b))) if b else 0.0
        if area_a + area_b == 0:
            continue
        inter = float(coco.area(coco.merge([coco.merge(a), coco.merge(b)], intersect=True))) if a and b else 0.0
        dices.append(2 * inter / (area_a + area_b))
    summary = aggregate(rows)
    return {"reference_instances": sum(map(len, ref.values())), "other_instances": sum(map(len, other.values())),
            "images_compared": len(rows), "matched_pq": summary["pq"], "tp": summary["tp"], "fp": summary["fp"],
            "fn": summary["fn"], "mean_pixel_dice": float(np.mean(dices)) if dices else 1.0}


def summary_table(metrics: dict) -> str:
    lines = [f"{'candidate':42s} {'calib PQ':>9s} {'val PQ':>8s} {'95% CI':>17s} {'TP/FP/FN':>16s} {'no-group':>9s}"]
    for cand, r in sorted(metrics["candidates"].items(), key=lambda kv: -kv[1]["calibration_pq"]):
        v = r["validation"]
        mark = " <- selected" if cand == metrics["selected_by_calibration"] else ""
        lines.append(f"{cand:42s} {r['calibration_pq']:9.4f} {v['pq']:8.4f} "
                     f"{v['pq_bootstrap_95ci'][0]:8.4f}-{v['pq_bootstrap_95ci'][1]:.4f} "
                     f"{v['tp']:>5d}/{v['fp']}/{v['fn']:<5d} {r['validation_pq_without_grouping']:9.4f}{mark}")
    return "\n".join(lines)


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    kind, job_path = argv[0], argv[1]
    job = json.loads(Path(job_path).read_text())
    {"train": _train_worker, "infer": _infer_worker}[kind](job)


if __name__ == "__main__":
    main()
