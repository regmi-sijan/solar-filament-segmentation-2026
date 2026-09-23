# Solar Filament Segmentation Challenge 2026

A reproducible native-resolution, CPU-based instance-segmentation baseline by Sijan Regmi. It uses Gaussian local contrast, solar-disk masking, morphological closing, and connected components. It does not require pretrained weights or external data.

## Setup

Tested with Python 3.14 on ARM64 macOS with 16 GiB RAM. Install the exact packages in a dedicated environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download the competition archive after accepting Kaggle's rules. Extract it at the repository root:

```
MAGFiLO_1.0_Kaggle_2026/
  train/train_images/
  train/MAGFiLO_1.0_Annotations_kaggle2026_train.json
  test/test_images/
```

Data source: https://www.kaggle.com/competitions/filament-segmentation-2026/data
The images and labels are not redistributed in this repository. Their CC BY-NC 4.0 terms are separate from source-code licensing.

## Reproduce the submitted predictions

```sh
python -m filament.run predict
python -m filament.run verify
```

The saved `configs/model.json` is the complete model: no weights, training service, network call, or other external file is required. `results/submission.csv` has `filament_id,segmentation_rle`; `results/inference-manifest.json` lists all processed images. Verification decodes and re-encodes each mask. An image with zero retained instances has no CSV row, not a fake mask.

## Reproduce parameter calibration and validation

```sh
python -m filament.run split
python -m filament.run calibrate-initial
python -m filament.run calibrate
python -m filament.run evaluate
python -m unittest discover -s tests -v
```

Calibration searches 32 initial and 64 refinement configurations on 111 images from the development partition. The 139-image validation partition is grouped by capture date and stratified by year. Do not tune on it after reviewing the reported result without designating it development data and creating a new holdout. Runtime depends on hardware; full calibration takes several minutes on the reference laptop. Commands accept `--data /path/to/MAGFiLO_1.0_Kaggle_2026`.

Run `python scripts/execute_notebook.py` to execute the supplied plain-Python notebook cells and save outputs without a local Jupyter server. The notebook also works interactively in Jupyter.

`notebooks/competition-pipeline.ipynb` demonstrates all stages and includes executed validation and inference. Recalibration is opt-in because the chosen parameters and all search results are already saved. The official self-evaluation notebook is retained locally as a reference, attributed to Azim Ahmadzadeh; our tests compare metric semantics against its definitions. It is not redistributed in the source release; the optional parity check skips when the reference notebook is absent.

## Report and figures

```sh
python scripts/make_figures.py
python scripts/build_report.py
cd report
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The LaTeX report follows the organizers' acmart/sigconf template and preserves its fixed subtitle, introduction, and acknowledgment. A TeX distribution with acmart is needed only to rebuild the PDF; it is not needed for inference. Template source: https://www.overleaf.com/read/vjztpvrnbrdh#43fbbe

## Outputs and limitations

Frozen-configuration validation: **PQ 0.1796**, with a date-cluster bootstrap 95% interval of **0.1602–0.1992**, on 139 source images and 222 annotation sets. This is a working baseline, not a state-of-the-art claim.

See `results/validation.json` for measured PQ and uncertainty, `results/calibration.json` for all search trials, and `results/submission-verification.json` for the CSV checksum. Local validation is not a Kaggle leaderboard result. No leaderboard score should be inferred from it.

See `docs/method-walkthrough.md` for an explanation of the method and its weaknesses, and `docs/requirements-checklist.md` for remaining external submission steps. AI assistance was used for implementation and documentation. The participant remains responsible for understanding and defending the method.

## Competition references

- https://www.kaggle.com/competitions/filament-segmentation-2026
- https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook
- https://doi.org/10.1038/s41597-024-03876-y

Public repository publication and the organizers' final form are separate actions. Files prepared locally do not establish a completed competition submission.
