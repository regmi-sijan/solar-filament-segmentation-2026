"""Synthetic stand-in for the competition data, used only for tests and smoke runs.

It mimics the real layout: grayscale JPEG "solar disks" with dark elongated filaments,
a COCO-style annotation JSON with several annotator records per image, and the same
folder structure and file names as ``MAGFiLO_1.0_Kaggle_2026``.
"""
import json
from pathlib import Path

import cv2
import numpy as np


def _curve(rng: np.random.Generator, size: int) -> np.ndarray:
    """A random quadratic Bezier curve inside the disk, as an (N, 2) int array."""
    center, radius = size / 2, size * 0.30
    points = []
    for _ in range(3):
        angle, dist = rng.uniform(0, 2 * np.pi), rng.uniform(0, radius)
        points.append((center + dist * np.cos(angle), center + dist * np.sin(angle)))
    p0, p1, p2 = (np.array(p) for p in points)
    t = np.linspace(0, 1, 40)[:, None]
    return ((1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2).astype(np.int32)


def _polygons(mask: np.ndarray) -> list:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1, 2).flatten().astype(float).tolist() for c in contours if len(c) >= 3]


def _make_image(rng: np.random.Generator, size: int):
    image = np.full((size, size), 12, np.uint8)
    cv2.circle(image, (size // 2, size // 2), int(size * 0.42), 150, -1)
    image = cv2.GaussianBlur(image, (0, 0), 1.5).astype(np.float32)
    shapes = []
    for _ in range(int(rng.integers(1, 4))):
        curve, width = _curve(rng, size), int(rng.integers(2, 4))
        layer = np.zeros((size, size), np.uint8)
        cv2.polylines(layer, [curve], False, 1, width)
        image[layer > 0] -= 80
        shapes.append(layer)
    image += rng.normal(0, 5, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8), shapes


def _annotator_view(rng: np.random.Generator, layer: np.ndarray) -> np.ndarray:
    """Jitter one filament mask the way a second annotator might: shift and grow/shrink a little."""
    dx, dy = int(rng.integers(-1, 2)), int(rng.integers(-1, 2))
    shifted = np.roll(layer, (dy, dx), axis=(0, 1))
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(shifted, kernel) if rng.random() < 0.5 else shifted


def make_synthetic_dataset(root, n_train: int = 12, n_test: int = 3, size: int = 128, seed: int = 0) -> dict:
    """Write a tiny dataset under ``root`` and return a matching ``split`` dictionary."""
    root = Path(root)
    (root / "train/train_images").mkdir(parents=True, exist_ok=True)
    (root / "test/test_images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    images, annotations, names = [], [], []
    image_id = annotation_id = 1
    for index in range(n_train):
        name = f"2015{index % 12 + 1:02d}{index % 27 + 1:02d}1200{index:02d}Mh.jpeg"
        image, shapes = _make_image(rng, size)
        cv2.imwrite(str(root / "train/train_images" / name), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        names.append(name)
        for _ in range(int(rng.integers(1, 4))):  # one record per annotator
            images.append({"id": image_id, "file_name": name, "height": size, "width": size})
            for layer in shapes:
                polygons = _polygons(_annotator_view(rng, layer))
                if polygons:
                    annotations.append({"id": annotation_id, "image_id": image_id, "segmentation": polygons})
                    annotation_id += 1
            image_id += 1
    for index in range(n_test):
        name = f"2016{index % 12 + 1:02d}{index % 27 + 1:02d}1300{index:02d}Mh.jpeg"
        image, _ = _make_image(rng, size)
        cv2.imwrite(str(root / "test/test_images" / name), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    path = root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    path.write_text(json.dumps({"images": images, "annotations": annotations}))
    n_val = max(1, n_train // 4)
    validation, rest = names[:n_val], names[n_val:]
    calibration = rest[:n_val]
    return {"validation": validation, "calibration": calibration, "training": rest}
