# Deep-learning extension (GPU U-Net)

Status: **implemented, smoke-tested on synthetic data, and run once end to end on the real competition data (see "First real-data run").**
Treat any later PQ figure as valid only if it comes from the `metrics.json` of that run, not from this document.

## First real-data run (Kaggle, Tesla T4, ImageNet-pretrained ResNet34, `TrainConfig()` defaults, seed 2026)

Trained on 457 images, tuned on the 111 calibration images, validation (139 images, 222 annotator sets) scored once.

| | Classical baseline | GPU U-Net |
|---|---|---|
| Held-out validation PQ | 0.1796 (95% CI 0.1602 to 0.1992) | **0.4060 (95% CI 0.3855 to 0.4248)** |
| TP / FP / FN | 444 / 1160 / 1114 | 914 / 563 / 644 |

* Best epoch by calibration PQ was epoch 40 of 40 (calibration-subset PQ 0.410); training took about 31 minutes (46 s per epoch).
* Post-processing chosen on calibration only: `low=0.6, high=0.9, min_area=600, max_area=150000, disk_erode=0, closing_radius=0`.
* The candidate CSV passed `verify_submission` (1245 rows, 180 test images, 177 with predictions; sha256 of the interactive-session file
  `33b61994929cb3395c97aef29be191d1a21f3da7921ab7f1197aad2c116aa14d`). It was **not** uploaded to Kaggle.
* Repeat run: saving the notebook as a Kaggle version ("Save & Run All", same code, same seed, 49 min 49 s on a T4) re-trained from scratch and gave
  **validation PQ 0.4111 (95% CI 0.3905 to 0.4308), TP / FP / FN 1047 / 813 / 511**, tuned post-processing `low=0.5, high=0.8, min_area=300, disk_erode=0`,
  and a candidate CSV of 1578 rows (178 of 180 test images with predictions; sha256
  `fc160d81172db64ae47711c3876fbf34e826442842f0760a6506e8f85215ec2b`). Run-to-run difference is about 0.005 PQ even with a fixed seed,
  because GPU kernels are not bit-deterministic and the calibration search can pick a different threshold.
  The files that persist are those of this second run (private notebook version, `run1/candidate_submission.csv`, `run1/best.pt`, `run1/metrics.json`).
* Caveats: two runs, one seed; the bootstrap interval reflects capture-date resampling of the 139 validation images, not seed variance;
  the baseline's local validation PQ (0.1796) was 0.15 on the public leaderboard, so expect some local-to-leaderboard gap.

## What changed relative to the classical baseline

| | Classical baseline | GPU extension |
|---|---|---|
| Pixel classifier | Gaussian local-contrast threshold | ResNet34 U-Net + full-resolution branch |
| Supervision | 6 scalar parameters fitted on 111 images | Training polygons of 457 images (soft consensus labels) |
| Resolution | 2048 px | 2048 px (trained on 512 px crops) |
| Instances | 8-connected components | 8-connected components with hysteresis threshold |
| Parameter selection | Calibration subset | Calibration subset (post-processing, best epoch) |
| Final score | Validation partition, once | Validation partition, once |

Everything downstream is shared with the baseline: `configs/split.json`, `core.score_masks` (strict IoU > 0.5, micro PQ),
the capture-date cluster bootstrap, and the CSV format.

## Protocol (no leakage)

* Train on `training` minus `calibration` (457 images). Capture dates never straddle partitions.
* Select the best epoch and tune threshold, hysteresis, minimum area and limb erosion on `calibration` (111 images).
* Score `validation` (139 images) exactly once. Do not tune on it afterwards; if you do, treat it as development data.
* Inputs are image pixels only. No spines, chirality labels, bounding boxes, external images or test labels are read.
  ImageNet-pretrained ResNet34 weights are optional (`TrainConfig.pretrained`); if used, name them in the report.

## Why soft labels

The metric scores each prediction against every annotator set of an image separately (1, 2 or 3 sets). The mean of the
annotators' masks is the best single target to regress, and a tuned threshold below 0.5 recovers the union where
annotators disagree.

## Run it

Kaggle GPU (recommended): import `notebooks/gpu-unet-training.ipynb`, attach the competition data, choose a GPU, *Run All*.
Local: `python -m filament.run_dl --data MAGFiLO_1.0_Kaggle_2026 --out results/dl` (add `--debug` for a quick check).
Outputs: `metrics.json`, `candidate_submission.csv`, `best.pt`, `validation_rows.json`, `inference_manifest.json`.
`candidate_submission.csv` is never uploaded by the code.

## Adopting it for the final entry

If the held-out PQ clearly beats 0.1796 (compare the bootstrap intervals, not just the point estimates):

1. Upload `candidate_submission.csv` to Kaggle manually and select it as a final submission.
2. Update the technical report (method, calibration protocol, pretrained-weights statement, PQ table) and keep it at four pages.
3. Copy the checkpoint hash, CSV hash and config into `docs/submission-status.json` and `release-manifest.json`; add `torchvision` to `requirements.txt`.
4. Re-file the organizers' Google form with the exact CSV and the new PDF before Nov 15, 2026, 11:59 pm CDT.
