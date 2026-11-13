# Final entry reproduction

The final selection is the 0.38 public-score submission from `sijanregmi/notebook556361ae1c`, version 3 (`359548383`). That notebook copies the CSV produced by `sijanregmi/notebook41c46ec4e4`, the tree-selection run. The accepted file is retained at `results/submission_final.csv` with SHA-256 `2a01f6e07ab24a52571e7fa0770cabe8ca1d95d5c191737acbb12f54564e3405`.

## Kaggle workflow

1. Import `notebooks/gpu-experiments-v2.ipynb`, attach the Solar Filament competition input, select two T4 GPUs, enable internet for the public ImageNet weights and package installation, and use Save & Run All. This creates `v2/suite_metrics.json` and the selected ResNet34/ResNet50 checkpoints under `v2/models/`. The earlier v1 checkpoint is optional; the selected final two-model ensemble does not include it.
2. Import `notebooks/gpu-refit-707.ipynb`. Attach the competition input and the saved output of step 1. Run on two T4 GPUs. This creates `refit/refit_metrics.json` and final EMA checkpoints under `refit/models/` using all 707 labeled images.
3. Import `notebooks/gpu-treesel.ipynb`. Attach the competition input and saved outputs of both preceding steps. Run on two T4 GPUs. It infers probability maps, fits the component-tree scorer, and writes `treesel/submission_refit_models.csv` for the final method. The held-out-model CSV is an experimental comparison.
4. Verify the generated CSV with the notebook's RLE/identifier checks. GPU training can vary between runs, so a newly trained set of weights can generate a different checksum. The archived accepted CSV preserves the exact scored predictions.

The notebooks embed the required project modules and split configuration. The repository includes their readable Python counterparts, package versions in `requirements.txt`, and the report's source and compiled PDF. Competition images must be attached through Kaggle. The original trained checkpoint files are retained in the saved Kaggle runs rather than in this Git repository; the notebooks provide the training path to regenerate them. No training was repeated during final-package preparation.

## Evaluation provenance

The networks used for developmental evaluation train on 457 images, with 111 calibration images and 139 validation images separated by capture date. The tree-selection method measured validation PQ 0.4404. Its final scorer is then fitted on calibration plus validation maps, and the selected final prediction file uses networks refitted on all 707 images. The final refit has no independent held-out score. The report explicitly describes the resulting change in probability-map distribution.
