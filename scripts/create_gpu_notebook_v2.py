"""Build notebooks/gpu-experiments-v2.ipynb: the v2 experiment suite as a self-contained Kaggle notebook.

The package files are written into the session with %%writefile (so the code stays readable and the
GPU worker processes can import it), and configs/split.json is embedded.

    python scripts/create_gpu_notebook_v2.py
"""
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]

INTRO = """# Solar filament segmentation: experiment suite v2 (GPU T4 x2)

v1 (one ResNet34 U-Net, 512 px crops) scored validation PQ 0.4111 and public leaderboard 0.33.
This notebook tests what should move that number, under the same leakage-free protocol:

* **Error analysis**: annotator-vs-annotator agreement (the human ceiling under this metric), how often a
  ground-truth filament consists of several disjoint pieces, and the PQ of the annotators' own consensus.
* **Stronger models, trained in parallel on both GPUs**: ResNet34 and ResNet50 U-Nets with 768 px crops
  (more context for long filaments) and an exponential moving average of the weights.
* **Instance formation**: merging nearby fragments into one filament and filling holes, tuned with the
  other post-processing settings.
* **Ensembles** of the new models and the v1 checkpoint (attach the v1 notebook's output as an input).

Every model trains on `training` minus `calibration`; checkpoint, post-processing and the choice between
candidates use `calibration` only; each candidate is scored once on `validation` and gets its own test CSV
in `/kaggle/working/v2/candidates/`. **Nothing is submitted to Kaggle by this notebook.**"""

SETUP = """import importlib.util, json, os, shutil, subprocess, sys, time
from pathlib import Path

if importlib.util.find_spec('pycocotools') is None:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pycocotools==2.0.11'])

import torch
INPUT_ROOT = Path(os.environ.get('FILAMENT_INPUT', '/kaggle/input'))
WORK_ROOT = Path(os.environ.get('FILAMENT_WORK', '/kaggle/working')).resolve()
os.chdir(WORK_ROOT)
Path('code/filament').mkdir(parents=True, exist_ok=True)
print('torch', torch.__version__, '| GPUs:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
      '| CPUs:', os.cpu_count())"""

IMPORT = """sys.path.insert(0, str(Path('code').resolve()))
from filament import suite
print('suite module:', suite.__file__)"""

LOCATE = """annotation_files = list(INPUT_ROOT.rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
assert len(annotation_files) == 1, f'Attach the competition data; found {len(annotation_files)} annotation files'
DATA = annotation_files[0].parent.parent
print('data root:', DATA)
v1_candidates = sorted(INPUT_ROOT.rglob('run1/best.pt'))
V1_WEIGHTS = v1_candidates[0] if v1_candidates else None
print('v1 checkpoint:', V1_WEIGHTS or 'not attached (ensembles with v1 are skipped)')"""

DEBUG = """# Every stage on a handful of images (both GPU workers, inference, tuning, CSV export): about 5 minutes.
debug_metrics = suite.run_suite(DATA, SPLIT, WORK_ROOT / 'debug_v2', suite.default_suite(V1_WEIGHTS, debug=True), debug=True)
print(suite.summary_table(debug_metrics))
shutil.rmtree(WORK_ROOT / 'debug_v2' / 'models', ignore_errors=True)  # tiny untrained checkpoints, not worth keeping
print('debug run finished: every stage ran (scores after 6 training steps are meaningless).')"""

FULL = """RUN_FULL = torch.cuda.is_available()   # the full suite needs GPUs
SUITE = suite.default_suite(V1_WEIGHTS)
print(json.dumps({m['name']: {k: m['cfg'][k] for k in ('encoder', 'crop', 'epochs', 'batch_size', 'ema_decay')}
                  for m in SUITE['models']}, indent=1))
if RUN_FULL:
    metrics = suite.run_suite(DATA, SPLIT, WORK_ROOT / 'v2', SUITE)
else:
    print('Skipped: no GPU available.')"""

SUMMARY = """if RUN_FULL:
    print(suite.summary_table(metrics))
    print()
    print('selected by calibration PQ:', metrics['selected_by_calibration'])
    print('diagnostics:', json.dumps(metrics['diagnostics'], indent=1))
    print('timings (min):', {k: round(v, 1) for k, v in metrics['timings'].items()})
    for cand, r in metrics['candidates'].items():
        print(cand, r['post'], r['csv']['rows'], 'rows', r['csv']['sha256'][:16])"""


def main() -> None:
    split = json.loads((ROOT / "configs/split.json").read_text())
    split_cell = "SPLIT = json.loads(r'''" + json.dumps(split, separators=(",", ":")) + "''')\n" \
        "print({key: len(value) for key, value in SPLIT.items() if isinstance(value, list)})"
    files = [
        ("code/filament/__init__.py", '"""Solar filament segmentation package (written by the notebook)."""\n'),
        ("code/filament/core.py", (ROOT / "filament/core.py").read_text()),
        ("code/filament/dl.py", (ROOT / "filament/dl.py").read_text()),
        ("code/filament/suite.py", (ROOT / "filament/suite.py").read_text()),
    ]
    cells = [nbformat.v4.new_markdown_cell(INTRO), nbformat.v4.new_code_cell(SETUP)]
    cells += [nbformat.v4.new_code_cell(f"%%writefile {path}\n{source.rstrip()}\n") for path, source in files]
    cells += [
        nbformat.v4.new_code_cell(IMPORT),
        nbformat.v4.new_code_cell(split_cell),
        nbformat.v4.new_code_cell(LOCATE),
        nbformat.v4.new_markdown_cell("## 1. Debug run of the whole suite (about 5 minutes)"),
        nbformat.v4.new_code_cell(DEBUG),
        nbformat.v4.new_markdown_cell("## 2. Full suite (about 2 to 2.5 hours on T4 x2)"),
        nbformat.v4.new_code_cell(FULL),
        nbformat.v4.new_code_cell(SUMMARY),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}
    nbformat.validate(notebook)
    out = ROOT / "notebooks/gpu-experiments-v2.ipynb"
    nbformat.write(notebook, out)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB, {len(cells)} cells)")


if __name__ == "__main__":
    main()
