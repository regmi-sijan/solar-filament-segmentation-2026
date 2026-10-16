"""Deep-learning extension: a soft-label U-Net trained on annotator polygons.

The classical baseline in ``core.py`` thresholds a local-contrast map. This module
replaces that pixel classifier with a ResNet34 U-Net trained only on the competition's
training polygons (no spines, chirality labels, external data, or test labels) and keeps
everything else identical: the date-grouped split in ``configs/split.json``, the micro-PQ
metric from ``core.score_masks``, and the CSV format written by ``run.predict``.

Design choices
--------------
* Native 2048 px resolution. Filaments are thin, so downscaling would cost IoU.
* Soft targets. Each image's target is the mean of its annotators' masks. The metric
  scores a prediction against every annotator set separately, so the consensus is the
  best single prediction to aim for.
* Hysteresis threshold, minimum area and limb erosion are tuned on the calibration
  subset only. The validation partition is evaluated once, at the end.
"""
import copy
import csv
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pycocotools import mask as coco
from scipy import ndimage

from .core import aggregate, encode_instances, ground_truth, load_annotations, score_masks


# --------------------------------------------------------------------------- configuration
@dataclass
class TrainConfig:
    """Training and inference settings for the U-Net."""

    crop: int = 512                 # training crop size in native pixels
    batch_size: int = 8
    epochs: int = 40
    steps_per_epoch: int = 150
    lr: float = 5e-4
    weight_decay: float = 1e-4
    positive_crop_prob: float = 0.7  # share of crops centred on an annotated pixel
    pretrained: bool = True          # ImageNet weights (needs internet; falls back to random)
    amp: bool = True                 # mixed precision on CUDA
    tta: bool = True                 # average over flips at inference
    workers: int = 3
    eval_every: int = 5              # epochs between quick tuning-set PQ checks
    eval_images: int = 40            # tuning images used for those quick checks
    seed: int = 2026
    encoder: str = "resnet34"        # see ENCODERS
    ema_decay: float = 0.0           # >0 keeps an exponential moving average of the weights


@dataclass
class PostConfig:
    """Turns a probability map into filament instances."""

    low: float = 0.5        # pixels above this form candidate regions
    high: float = 0.5       # a region is kept only if its peak probability reaches this
    min_area: int = 300
    max_area: int = 150000
    disk_erode: int = 10    # pixels removed from the solar limb
    closing_radius: int = 0
    group_dist: int = 0     # >0 merges fragments closer than about 2*group_dist px into one filament
    fill_holes: bool = False


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- data
def image_stats(img: np.ndarray) -> tuple:
    """Mean and standard deviation of the non-black pixels (the solar disk)."""
    pixels = img[img > 8].astype(np.float32)
    if pixels.size < 100:
        pixels = img.astype(np.float32).ravel()
    return float(pixels.mean()), float(pixels.std() + 1e-3)


def normalize(img: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (img.astype(np.float32) - mean) / std


def solar_disk(img: np.ndarray) -> np.ndarray:
    """Filled solar-disk mask (uint8 0/1) from Otsu thresholding, as in core.contrast_maps."""
    _, bright = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
    if count < 2:
        return np.ones(img.shape, np.uint8)
    mask = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask[:] = 0
    cv2.drawContours(mask, contours, -1, 1, cv2.FILLED)
    return mask


def erode_disk(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.erode(mask, kernel)


def load_case(data_dir, name: str, records: dict, annotations: dict) -> dict:
    """Read one training image with its soft consensus target and per-annotator masks."""
    img = cv2.imread(str(Path(data_dir) / "train/train_images" / name), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read training image {name}")
    gt = ground_truth(records[name], annotations)
    target = np.zeros(img.shape, np.float32)
    for _, masks in gt:
        if masks:
            target += coco.decode(coco.merge(masks))
    target /= max(1, len(gt))
    target_u8 = np.round(target * 255).astype(np.uint8)
    positives = np.flatnonzero(target_u8.ravel() >= 128)
    if positives.size > 4096:
        positives = np.random.default_rng(0).choice(positives, 4096, replace=False)
    mean, std = image_stats(img)
    return {"name": name, "image": img, "target": target_u8, "gt": gt, "disk": solar_disk(img),
            "mean": mean, "std": std, "positives": positives.astype(np.int64)}


def load_cases(data_dir, names, records, annotations, log=print) -> list:
    cases = []
    for index, name in enumerate(names, 1):
        cases.append(load_case(data_dir, name, records, annotations))
        if index % 50 == 0 or index == len(names):
            log(f"loaded {index}/{len(names)} images")
    return cases


def sample_crop(case: dict, size: int, rng: np.random.Generator, positive_prob: float):
    """Random crop, centred near an annotated pixel with probability ``positive_prob``."""
    img, target = case["image"], case["target"]
    height, width = img.shape
    if case["positives"].size and rng.random() < positive_prob:
        flat = int(case["positives"][rng.integers(case["positives"].size)])
        center_y, center_x = divmod(flat, width)
        y0 = center_y - int(rng.integers(size // 8, size - size // 8))
        x0 = center_x - int(rng.integers(size // 8, size - size // 8))
    else:
        y0 = int(rng.integers(0, max(1, height - size + 1)))
        x0 = int(rng.integers(0, max(1, width - size + 1)))
    y0 = int(np.clip(y0, 0, max(0, height - size)))
    x0 = int(np.clip(x0, 0, max(0, width - size)))
    patch = normalize(img[y0:y0 + size, x0:x0 + size], case["mean"], case["std"])
    soft = target[y0:y0 + size, x0:x0 + size].astype(np.float32) / 255.0
    if patch.shape != (size, size):
        pad = ((0, size - patch.shape[0]), (0, size - patch.shape[1]))
        patch = np.pad(patch, pad, constant_values=-case["mean"] / case["std"])
        soft = np.pad(soft, pad)
    return patch, soft


def augment(patch: np.ndarray, soft: np.ndarray, rng: np.random.Generator):
    """Dihedral flips plus brightness, contrast and noise jitter."""
    if rng.random() < 0.5:
        patch, soft = patch[:, ::-1], soft[:, ::-1]
    if rng.random() < 0.5:
        patch, soft = patch[::-1], soft[::-1]
    if rng.random() < 0.5:
        patch, soft = patch.T, soft.T
    patch = patch * rng.uniform(0.8, 1.25) + rng.uniform(-0.2, 0.2)
    patch = patch + rng.normal(0.0, rng.uniform(0.0, 0.05), patch.shape)
    return patch.astype(np.float32), soft


class CropDataset(torch.utils.data.Dataset):
    """Endless stream of random augmented crops; deterministic given (seed, epoch, index)."""

    def __init__(self, cases: list, cfg: TrainConfig, length: int):
        self.cases, self.cfg, self.length, self.epoch = cases, cfg, length, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        rng = np.random.default_rng([self.cfg.seed, self.epoch, index])
        case = self.cases[int(rng.integers(len(self.cases)))]
        patch, soft = sample_crop(case, self.cfg.crop, rng, self.cfg.positive_crop_prob)
        patch, soft = augment(patch, soft, rng)
        return (torch.from_numpy(np.ascontiguousarray(patch))[None],
                torch.from_numpy(np.ascontiguousarray(soft, dtype=np.float32))[None])


# --------------------------------------------------------------------------- model
def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    )


class UpBlock(nn.Module):
    """Upsample, concatenate the encoder skip connection, and convolve twice."""

    def __init__(self, c_in: int, c_skip: int, c_out: int):
        super().__init__()
        self.conv = conv_block(c_in + c_skip, c_out)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# name -> (torchvision builder, weights enum, channels of stem, layer1, layer2, layer3, layer4)
ENCODERS = {
    "resnet34": ("resnet34", "ResNet34_Weights", (64, 64, 128, 256, 512)),
    "resnet50": ("resnet50", "ResNet50_Weights", (64, 256, 512, 1024, 2048)),
    "resnext50": ("resnext50_32x4d", "ResNeXt50_32X4D_Weights", (64, 256, 512, 1024, 2048)),
}


class FilamentUNet(nn.Module):
    """ResNet encoder, U-Net decoder, and a shallow branch at full input resolution.

    The ResNet stem halves the resolution immediately, which blurs thin barbs; the extra
    full-resolution branch gives the head direct access to the raw pixels. With the default
    ``encoder="resnet34"`` the layer names and shapes are those of the v1 model, so v1
    checkpoints load unchanged.
    """

    def __init__(self, pretrained: bool = True, encoder: str = "resnet34"):
        super().__init__()
        import torchvision

        builder_name, weights_name, (c0, c1, c2, c3, c4) = ENCODERS[encoder]
        builder = getattr(torchvision.models, builder_name)
        weights = getattr(torchvision.models, weights_name).DEFAULT if pretrained else None
        try:
            net = builder(weights=weights)
        except Exception as error:  # offline sessions cannot download the weights
            print(f"Pretrained weights unavailable ({type(error).__name__}); using random init.")
            net = builder(weights=None)
        conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            conv1.weight.copy_(net.conv1.weight.sum(dim=1, keepdim=True))
        self.stem = nn.Sequential(conv1, net.bn1, net.relu)
        self.pool = net.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = net.layer1, net.layer2, net.layer3, net.layer4
        self.up4 = UpBlock(c4, c3, 256)
        self.up3 = UpBlock(256, c2, 128)
        self.up2 = UpBlock(128, c1, 64)
        self.up1 = UpBlock(64, c0, 64)
        self.full = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(64 + 16, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x):
        height, width = x.shape[-2:]
        pad_h, pad_w = (-height) % 32, (-width) % 32
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        full = self.full(x)
        stem = self.stem(x)
        e1 = self.layer1(self.pool(stem))
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        d = self.up4(e4, e3)
        d = self.up3(d, e2)
        d = self.up2(d, e1)
        d = self.up1(d, stem)
        d = F.interpolate(d, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(torch.cat([d, full], dim=1))[..., :height, :width]


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Binary cross-entropy plus batch soft-Dice; both accept soft targets."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum()
    dice = 1.0 - (2.0 * inter + eps) / (prob.sum() + target.sum() + eps)
    return bce + dice


# --------------------------------------------------------------------------- inference
@torch.no_grad()
def predict_proba(model: nn.Module, img: np.ndarray, device: torch.device, cfg: TrainConfig,
                  tta=None) -> np.ndarray:
    """Probability map (float16, native resolution), optionally averaged over flips."""
    use_tta = cfg.tta if tta is None else tta
    use_amp = cfg.amp and device.type == "cuda"
    model.eval()
    mean, std = image_stats(img)
    x = torch.from_numpy(normalize(img, mean, std))[None, None].to(device)
    flips = [(), (3,), (2,), (2, 3)] if use_tta else [()]
    total = None
    for dims in flips:
        inp = torch.flip(x, dims) if dims else x
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model(inp)
        out = torch.sigmoid(out.float())
        if dims:
            out = torch.flip(out, dims)
        total = out if total is None else total + out
    return (total / len(flips))[0, 0].cpu().numpy().astype(np.float16)


def fill_holes(binary: np.ndarray) -> np.ndarray:
    """Fill background regions not 4-connected to the border (= scipy binary_fill_holes), fast."""
    height, width = binary.shape
    flood = np.zeros((height + 2, width + 2), np.uint8)
    flood[1:-1, 1:-1] = binary > 0
    cv2.floodFill(flood, np.zeros((height + 4, width + 4), np.uint8), (0, 0), 1, flags=4)
    return ((binary > 0) | (flood[1:-1, 1:-1] == 0)).astype(np.uint8)


def segment(prob: np.ndarray, disk: np.ndarray, post: PostConfig):
    """Probability map -> (label image, accepted labels), the format encode_instances expects.

    Steps: threshold at ``low`` inside the (eroded) solar disk, optionally fill holes and
    close, label 8-connected regions, optionally merge regions closer than ~2*group_dist px
    (one filament is often annotated as several disjoint pieces), then keep instances whose
    area is in range and whose peak probability reaches ``high``. With ``group_dist=0`` and
    ``fill_holes=False`` this is exactly the v1 rule.
    """
    binary = ((prob > post.low) & (erode_disk(disk, post.disk_erode) > 0)).astype(np.uint8)
    if post.fill_holes:
        binary = fill_holes(binary)
    if post.closing_radius:
        size = post.closing_radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    count, labels = cv2.connectedComponents(binary, connectivity=8, ltype=cv2.CV_32S)
    if post.group_dist > 0 and count > 2:
        size = 2 * post.group_dist + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        count, grouped = cv2.connectedComponents(cv2.dilate(binary, kernel), connectivity=8, ltype=cv2.CV_32S)
        labels = np.where(binary > 0, grouped, 0).astype(np.int32)
    if count <= 1:
        return labels, []
    index = np.arange(1, count)
    areas = np.bincount(labels.ravel(), minlength=count)[1:]
    keep = (areas >= post.min_area) & (areas <= post.max_area)
    if post.high > post.low and keep.any():
        peaks = np.asarray(ndimage.maximum(prob.astype(np.float32), labels, index[keep]))
        keep[np.flatnonzero(keep)] = peaks >= post.high
    return labels, index[keep].tolist()


def encode_instances_fast(labels: np.ndarray, accepted: list) -> list:
    """Same RLEs as core.encode_instances, but reorders the label image to Fortran order once."""
    if not accepted:
        return []
    fortran = np.asfortranarray(labels)
    return [coco.encode((fortran == label).astype(np.uint8)) for label in accepted]


# --------------------------------------------------------------------------- evaluation
def count_matches(gt_masks: list, pred: list) -> dict:
    """tp/fp/fn/sum_iou with the same strict IoU > 0.5 rule as core.score_masks, but light."""
    n_gt, n_pred = len(gt_masks), len(pred)
    if not (n_gt and n_pred):
        return {"tp": 0, "fp": n_pred, "fn": n_gt, "sum_iou": 0.0}
    iou = coco.iou(pred, gt_masks, [0] * n_gt).T
    hit = iou > 0.5
    return {"tp": int(hit.sum()), "fp": int((hit.sum(axis=0) == 0).sum()),
            "fn": int((hit.sum(axis=1) == 0).sum()), "sum_iou": float(iou[hit].sum())}


def evaluate_probs(cases: list, probs: dict, post: PostConfig, full: bool = False) -> list:
    """One row per annotator record, exactly as run.evaluate builds them."""
    rows = []
    for case in cases:
        labels, accepted = segment(probs[case["name"]], case["disk"], post)
        pred = encode_instances(labels, accepted)
        for record_id, masks in case["gt"]:
            score = score_masks(masks, pred) if full else count_matches(masks, pred)
            rows.append({"file_name": case["name"], "image_record": record_id, **score})
    return rows


def bootstrap_ci(rows: list, samples: int = 2000, seed: int = 2026):
    """Capture-date cluster bootstrap of micro PQ, as in run.evaluate."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["file_name"][:8]].append(row)
    clusters = [aggregate(group) for group in groups.values()]
    rng = np.random.default_rng(seed)
    pqs = [aggregate([clusters[i] for i in rng.integers(0, len(clusters), len(clusters))])["pq"]
           for _ in range(samples)]
    return np.percentile(pqs, [2.5, 97.5]).tolist(), len(groups)


def tune_post(cases: list, probs: dict, start: PostConfig, log=print) -> PostConfig:
    """Coordinate search over post-processing settings, using the tuning subset only."""
    best = start
    best_pq = aggregate(evaluate_probs(cases, probs, best))["pq"]
    log(f"post start pq={best_pq:.4f} {asdict(best)}")

    def trial(**changes):
        nonlocal best, best_pq
        candidate = replace(best, **changes)
        pq = aggregate(evaluate_probs(cases, probs, candidate))["pq"]
        log(f"post trial {changes} pq={pq:.4f}")
        if pq > best_pq:
            best, best_pq = candidate, pq

    for threshold in (0.3, 0.4, 0.5, 0.6):
        trial(low=threshold, high=threshold)
    for area in (50, 150, 300, 600):
        trial(min_area=area)
    for margin in (0.0, 0.15, 0.3):
        trial(high=min(0.95, best.low + margin))
    for radius in (0, 10, 20):
        trial(disk_erode=radius)
    log(f"post best pq={best_pq:.4f} {asdict(best)}")
    return best


# --------------------------------------------------------------------------- training
class ModelEMA:
    """Exponential moving average of all weights and BatchNorm statistics.

    The decay ramps up as (1 + step) / (10 + step) so early, random weights fade quickly.
    """

    def __init__(self, model: nn.Module, decay: float):
        self.decay, self.steps = decay, 0
        self.shadow = {k: v.detach().clone().float() if v.dtype.is_floating_point else v.detach().clone()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.steps += 1
        decay = min(self.decay, (1 + self.steps) / (10 + self.steps))
        for key, value in model.state_dict().items():
            if value.dtype.is_floating_point:
                self.shadow[key].mul_(decay).add_(value.detach().float(), alpha=1.0 - decay)
            else:
                self.shadow[key].copy_(value)

    def state_dict(self) -> dict:
        return {k: v.clone() for k, v in self.shadow.items()}


def train_model(model, train_cases, tune_cases, cfg: TrainConfig, device, out_dir, log=print) -> list:
    """Train with BCE+Dice, keep the checkpoint with the best quick tuning-set PQ.

    With ``cfg.ema_decay > 0`` the moving-average weights are the ones evaluated and saved.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ema = ModelEMA(model, cfg.ema_decay) if cfg.ema_decay > 0 else None
    eval_model = copy.deepcopy(model) if ema else model
    use_amp = cfg.amp and device.type == "cuda"
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, total_steps=cfg.epochs * cfg.steps_per_epoch, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    dataset = CropDataset(train_cases, cfg, cfg.steps_per_epoch * cfg.batch_size)
    eval_cases = tune_cases[: cfg.eval_images]
    best_pq, history = -1.0, []
    for epoch in range(cfg.epochs):
        dataset.set_epoch(epoch)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.batch_size, num_workers=cfg.workers,
            pin_memory=device.type == "cuda", drop_last=True)
        model.train()
        start, losses = time.time(), []
        for patch, soft in loader:
            patch, soft = patch.to(device, non_blocking=True), soft.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(patch)
            loss = bce_dice_loss(logits.float(), soft)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema:
                ema.update(model)
            losses.append(float(loss.detach()))
        record = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "seconds": time.time() - start}
        if (epoch + 1) % cfg.eval_every == 0 or epoch + 1 == cfg.epochs:
            if ema:
                eval_model.load_state_dict(ema.state_dict())
            probs = {c["name"]: predict_proba(eval_model, c["image"], device, cfg, tta=False) for c in eval_cases}
            record["tune_pq"] = aggregate(evaluate_probs(eval_cases, probs, PostConfig()))["pq"]
            if record["tune_pq"] > best_pq:
                best_pq = record["tune_pq"]
                torch.save(eval_model.state_dict(), out_dir / "best.pt")
        history.append(record)
        log(record)
    final = ema.state_dict() if ema else model.state_dict()
    torch.save(final, out_dir / "last.pt")
    if not (out_dir / "best.pt").exists():
        torch.save(final, out_dir / "best.pt")
    return history


# --------------------------------------------------------------------------- submission files
def list_images(directory) -> list:
    return sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))


def write_submission(model, test_dir, post: PostConfig, cfg: TrainConfig, device, out_csv, log=print,
                     limit=None) -> dict:
    """Write one CSV row per predicted filament: ``filament_id,segmentation_rle`` (counts only)."""
    paths = list_images(test_dir)[:limit]
    manifest, start = [], time.time()
    with Path(out_csv).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filament_id", "segmentation_rle"])
        for index, path in enumerate(paths, 1):
            tick = time.time()
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            labels, accepted = segment(predict_proba(model, img, device, cfg), solar_disk(img), post)
            instances = encode_instances(labels, accepted)
            for number, rle in enumerate(instances, 1):
                writer.writerow([f"{path.stem}_{number}", rle["counts"].decode("ascii")])
            manifest.append({"file_name": path.name, "instances": len(instances), "seconds": time.time() - tick})
            if index % 20 == 0:
                log(f"predicted {index}/{len(paths)}")
    return {"seconds": time.time() - start, "images": manifest}


def verify_submission(csv_path, test_dir) -> dict:
    """Decode and re-encode every mask; check headers, unique ids, and image coverage."""
    paths = {p.stem: p for p in list_images(test_dir)}
    sizes, seen, ids, count = {}, set(), set(), 0
    with Path(csv_path).open() as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError(f"Unexpected CSV header: {reader.fieldnames}")
        for row in reader:
            fid = row["filament_id"]
            if fid in ids:
                raise ValueError(f"Duplicate filament_id {fid}")
            ids.add(fid)
            name, number = fid.rsplit("_", 1)
            if name not in paths or int(number) < 1:
                raise ValueError(f"Bad filament_id {fid}")
            if name not in sizes:
                sizes[name] = cv2.imread(str(paths[name]), cv2.IMREAD_GRAYSCALE).shape
            counts = row["segmentation_rle"]
            if '"' in counts or "'" in counts:
                raise ValueError(f"Quote character in RLE for {fid}")
            rle = {"size": list(sizes[name]), "counts": counts.encode("ascii")}
            mask = coco.decode(rle)
            if not mask.any():
                raise ValueError(f"Empty mask for {fid}")
            if coco.encode(np.asfortranarray(mask))["counts"] != rle["counts"]:
                raise ValueError(f"RLE is not canonical for {fid}")
            seen.add(name)
            count += 1
    digest = hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()
    return {"status": "passed", "rows": count, "test_images": len(paths),
            "images_with_predictions": len(seen), "sha256": digest}


# --------------------------------------------------------------------------- experiment
def debug_overrides(cfg: TrainConfig) -> TrainConfig:
    """Tiny settings for a quick end-to-end check of the whole pipeline."""
    return replace(cfg, crop=256, batch_size=2, epochs=2, steps_per_epoch=4, workers=0,
                   eval_every=1, eval_images=2, tta=False, pretrained=False)


def run_experiment(data_dir, split: dict, out_dir, cfg=None, debug: bool = False,
                   baseline_pq: float = 0.1796, log=print) -> dict:
    """Train, tune on the calibration subset, evaluate once on validation, write a candidate CSV.

    Nothing is submitted to Kaggle. The candidate CSV and metrics are written to ``out_dir``.
    """
    cfg = debug_overrides(cfg or TrainConfig()) if debug else (cfg or TrainConfig())
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.seed)
    log(f"device={device} debug={debug}")

    calibration = set(split["calibration"])
    train_names = [n for n in split["training"] if n not in calibration]
    tune_names, val_names = list(split["calibration"]), list(split["validation"])
    test_dir = data_dir / "test/test_images"
    if debug:
        train_names, tune_names, val_names = train_names[:6], tune_names[:4], val_names[:4]
    log(f"train={len(train_names)} tune={len(tune_names)} validation={len(val_names)}")

    records, annotations = load_annotations(data_dir)
    train_cases = load_cases(data_dir, train_names, records, annotations, log)
    tune_cases = load_cases(data_dir, tune_names, records, annotations, log)

    model = FilamentUNet(cfg.pretrained, cfg.encoder).to(device)
    history = train_model(model, train_cases, tune_cases, cfg, device, out_dir, log)
    del train_cases
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))

    log("tuning post-processing on the calibration subset")
    tune_probs = {c["name"]: predict_proba(model, c["image"], device, cfg) for c in tune_cases}
    post = tune_post(tune_cases, tune_probs, PostConfig(), log)
    tune_pq = aggregate(evaluate_probs(tune_cases, tune_probs, post))["pq"]
    del tune_cases, tune_probs

    log("evaluating once on the held-out validation partition")
    val_cases = load_cases(data_dir, val_names, records, annotations, log)
    val_probs = {c["name"]: predict_proba(model, c["image"], device, cfg) for c in val_cases}
    rows = evaluate_probs(val_cases, val_probs, post, full=True)
    summary = aggregate(rows)
    interval, clusters = bootstrap_ci(rows)
    summary.update({"pq_bootstrap_95ci": interval, "images": len(val_cases),
                    "annotation_sets": len(rows), "date_clusters": clusters})
    log(f"validation PQ {summary['pq']:.4f} (baseline {baseline_pq:.4f}) 95% CI {interval}")

    log("writing the candidate submission (not uploaded anywhere)")
    csv_path = out_dir / "candidate_submission.csv"
    manifest = write_submission(model, test_dir, post, cfg, device, csv_path, log, limit=4 if debug else None)
    verification = verify_submission(csv_path, test_dir)
    metrics = {"validation": summary, "tuning_pq": tune_pq, "baseline_validation_pq": baseline_pq,
               "post": asdict(post), "train_config": asdict(cfg), "history": history,
               "verification": verification}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "validation_rows.json").write_text(json.dumps(rows))
    (out_dir / "inference_manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"done: {verification}")
    return metrics
