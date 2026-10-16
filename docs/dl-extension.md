# Deep-learning extension (GPU U-Net)

Status: **implemented, tested, and run on the real competition data: v1 (one ResNet34 U-Net, validation PQ 0.4111, public 0.33) and
the v2 suite (ensembles, validation PQ 0.4271 for the candidate selected on calibration).**
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
* Public leaderboard: the repeat-run CSV was submitted once as an exploratory (non-final) submission and scored **0.33**.

## Experiment suite v2 (Kaggle, T4 x2, `notebooks/gpu-experiments-v2.ipynb`, 2 h 45 min)

Same partitions and rules: every model trains on the 457 training-minus-calibration images; checkpoint, post-processing and
the choice between candidates use the 111 calibration images only; each candidate is then scored once on validation
(139 images, 222 annotator sets). Code: `filament/suite.py`; private Kaggle notebook `notebooke5e77a5780`, version 1.

| Candidate | Calibration PQ | Validation PQ (95% CI) | TP / FP / FN | Val. PQ without grouping |
|---|---|---|---|---|
| **Ensemble ResNet34-768 + ResNet50-768 (selected on calibration)** | **0.4240** | **0.4271 (0.4053 to 0.4466)** | 1033 / 689 / 525 | 0.4226 |
| Ensemble of both + v1 | 0.4224 | 0.4247 (0.4037 to 0.4448) | 996 / 626 / 562 | 0.4215 |
| ResNet50 U-Net, 768 px crops, EMA | 0.4196 | 0.4295 (0.4088 to 0.4489) | 1020 / 644 / 538 | 0.4238 |
| ResNet34 U-Net, 768 px crops, EMA | 0.4166 | 0.4208 (0.4006 to 0.4405) | 1015 / 674 / 543 | 0.4124 |
| v1 checkpoint, re-tuned | 0.4070 | 0.4163 (0.3953 to 0.4362) | 1014 / 704 / 544 | 0.4090 |

* Selected candidate (highest calibration PQ): the two-model ensemble, post-processing `low=0.6, high=0.9, min_area=300,
  disk_erode=10, group_dist=4, fill_holes=False`. Validation PQ 0.4271 vs 0.4111 for the v1 CSV that scored 0.33 publicly.
  ResNet50 alone scored slightly higher on validation (0.4295) but lower on calibration; the difference is within noise, and
  choosing it after seeing validation would leak.
* Every new candidate beats v1 under identical scoring; the gain is modest (+0.01 to +0.02 PQ) and the confidence intervals overlap.
* Instance grouping (merging fragments within ~2*group_dist px) adds about +0.005 PQ; the tuned thresholds are more conservative
  than v1's (low 0.6 to 0.7, high 0.85 to 0.95), trading a few true positives for fewer false positives.
* Candidate CSVs (all verified, none uploaded by the code) are in the notebook output under `v2/candidates/<candidate>/submission.csv`;
  sha256 prefixes: selected ensemble `392c9f61f9410c1f` (1436 rows), ResNet50 `6c7e80b4c9cd806d` (1411 rows),
  ResNet34 `fb92e53aca5745f1` (1394 rows), three-model ensemble `65f824e5d2dd508b` (1353 rows), v1 re-tuned `749ba9591e320c23` (1449 rows).

### Error analysis (from the same run)

* **Annotators disagree a lot under this metric.** Scoring one annotator's filaments against another's (101 calibration and
  validation images with two or more annotators) gives PQ **0.353**. All models above exceed that, because they predict the
  annotators' consensus, which is the best single answer when every annotator set is scored separately.
* Using the consensus mask itself as the prediction reaches PQ 0.78 on calibration (an upper reference, inflated by
  single-annotator images, where the consensus is the answer).
* Ground-truth filaments are single polygons (0 of 8199 annotations have several) and 0.3% have disconnected parts, so large
  grouping distances hurt (consensus PQ 0.78 -> 0.68 at 32 px); small distances help predictions whose filaments break up.

### Next step if this configuration is adopted

`filament.suite.run_refit` retrains the selected models on all 707 labelled images (training + validation), reuses the
calibration-tuned post-processing, and writes one CSV. No held-out data remains, so the quality estimate stays the
validation figure above.

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
