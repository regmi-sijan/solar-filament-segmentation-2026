"""Build notebooks/gpu-unet-training.ipynb: a self-contained Kaggle GPU notebook.

The notebook inlines filament/core.py and filament/dl.py (so it needs no repository
checkout on Kaggle) and embeds configs/split.json.

    python scripts/create_gpu_notebook.py
"""
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]

INTRO = """# Solar filament segmentation: GPU U-Net (soft labels, native resolution)

Deep-learning extension of the CPU baseline (local validation PQ 0.1796). A ResNet34 U-Net with a
full-resolution branch is trained on the training polygons only (no spines, chirality labels,
external annotations, or test labels). Post-processing is tuned on the calibration subset of
`configs/split.json`, the held-out validation partition is scored exactly once, and a candidate CSV
is written. **Nothing is submitted to Kaggle by this notebook.**

How to run:
1. *Add Input* -> the `filament-segmentation-2026` competition data.
2. *Session options* -> Accelerator: a GPU (T4 or P100). Turn Internet on if you want ImageNet-pretrained
   weights; without it the notebook trains from random initialisation and says so.
3. *Run All*. The first full cell is a 2-minute debug run; the second is the real run.

Outputs land in `/kaggle/working/run1/`: `metrics.json`, `candidate_submission.csv`, `best.pt`."""

SETUP = """import importlib.util, json, os, subprocess, sys
from pathlib import Path

if importlib.util.find_spec('pycocotools') is None:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pycocotools==2.0.11'])

import torch
INPUT_ROOT = Path(os.environ.get('FILAMENT_INPUT', '/kaggle/input'))
WORK_ROOT = Path(os.environ.get('FILAMENT_WORK', '/kaggle/working'))
print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
else:
    print('WARNING: no GPU. Only the debug run will execute; enable a GPU accelerator for the full run.')"""

LOCATE = """annotation_files = list(INPUT_ROOT.rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
assert len(annotation_files) == 1, f'Attach the competition data; found {len(annotation_files)} annotation files'
DATA = annotation_files[0].parent.parent
assert (DATA / 'train/train_images').is_dir() and (DATA / 'test/test_images').is_dir(), sorted(p.name for p in DATA.iterdir())
print('data root:', DATA)
print('train images:', len(list((DATA / 'train/train_images').iterdir())), '| test images:', len(list((DATA / 'test/test_images').iterdir())))"""

DEBUG = """# Quick end-to-end check on a handful of images: confirms data paths, the model, tuning, scoring and CSV export.
debug_metrics = run_experiment(DATA, SPLIT, WORK_ROOT / 'debug', debug=True)
print('debug run finished: every stage ran. (PQ after 8 training steps on 4 images is meaningless, often 0.)')"""

FULL = """RUN_FULL = torch.cuda.is_available()   # set True to force a (very slow) CPU run
CFG = TrainConfig()                     # crop 512, batch 8, 40 epochs x 150 steps; edit here to experiment
if RUN_FULL:
    metrics = run_experiment(DATA, SPLIT, WORK_ROOT / 'run1', CFG, baseline_pq=0.1796)
else:
    print('Skipped: no GPU available.')"""

SUMMARY = """if RUN_FULL:
    val = metrics['validation']
    print(f"held-out validation PQ: {val['pq']:.4f}  (95% CI {val['pq_bootstrap_95ci'][0]:.4f} to {val['pq_bootstrap_95ci'][1]:.4f})")
    print(f"classical baseline PQ:  {metrics['baseline_validation_pq']:.4f}")
    print(f"tuned post-processing:  {metrics['post']}")
    print(f"tp/fp/fn: {val['tp']} / {val['fp']} / {val['fn']}   images: {val['images']}   annotation sets: {val['annotation_sets']}")
    print('candidate CSV:', metrics['verification'])
    for record in metrics['history'][-5:]:
        print(record)"""


def inline(path: Path) -> str:
    """Module source without relative imports, so it can live in a notebook cell."""
    lines = [line for line in path.read_text().splitlines() if not line.startswith("from .")]
    return "\n".join(lines)


def main() -> None:
    split = json.loads((ROOT / "configs/split.json").read_text())
    split_cell = "SPLIT = json.loads(r'''" + json.dumps(split, separators=(",", ":")) + "''')\n" \
        "print({key: len(value) for key, value in SPLIT.items() if isinstance(value, list)})"
    code_cell = inline(ROOT / "filament/core.py") + "\n\n\n" + inline(ROOT / "filament/dl.py")
    cells = [
        nbformat.v4.new_markdown_cell(INTRO),
        nbformat.v4.new_code_cell(SETUP),
        nbformat.v4.new_code_cell(code_cell),
        nbformat.v4.new_code_cell(split_cell),
        nbformat.v4.new_code_cell(LOCATE),
        nbformat.v4.new_markdown_cell("## 1. Debug run (about 2 minutes)"),
        nbformat.v4.new_code_cell(DEBUG),
        nbformat.v4.new_markdown_cell("## 2. Full GPU run (train, tune on calibration, score validation once, write candidate CSV)"),
        nbformat.v4.new_code_cell(FULL),
        nbformat.v4.new_code_cell(SUMMARY),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}
    nbformat.validate(notebook)
    out = ROOT / "notebooks/gpu-unet-training.ipynb"
    nbformat.write(notebook, out)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB, {len(cells)} cells)")


if __name__ == "__main__":
    main()
