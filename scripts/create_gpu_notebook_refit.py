"""Build notebooks/gpu-refit-707.ipynb: retrain the v2 selection on all 707 labelled images (Kaggle T4 x2).

Inputs on Kaggle: the competition data and the output of the v2 suite notebook (for suite_metrics.json,
which fixes the selected models, their exact training settings and the calibration-tuned post-processing,
and for the v2 CSV used in the agreement check).

    python scripts/create_gpu_notebook_refit.py
"""
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]

INTRO = """# Solar filament segmentation: refit the v2 selection on all 707 labelled images

The v2 suite chose, on the calibration images only, an ensemble of a ResNet34 and a ResNet50 U-Net
(768 px crops, EMA weights). On the held-out validation images it scored PQ 0.4271 (95% CI 0.4053 to 0.4466),
and its test CSV scored 0.35 on the public leaderboard.

This notebook retrains exactly those two models on **all** labelled images (`training` + `validation` =
707 images instead of 457), with the same training settings, the same post-processing (read from the v2 run's
`suite_metrics.json`, not re-tuned), and the final EMA weights (no held-out data is left for checkpoint picking).
Its quality estimate therefore remains the v2 validation score. As a sanity check, the new test predictions are
compared with the v2 ensemble's CSV.

**Nothing is submitted to Kaggle by this notebook.** The CSV is written to `/kaggle/working/refit/submission.csv`."""

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
v2_metrics_files = sorted(INPUT_ROOT.rglob('v2/suite_metrics.json'))
assert v2_metrics_files, 'Attach the output of the v2 suite notebook (it holds v2/suite_metrics.json)'
V2 = json.loads(v2_metrics_files[0].read_text())
SELECTED = V2['selected_by_calibration']
MEMBERS = V2['candidates'][SELECTED]['members']
POST = V2['candidates'][SELECTED]['post']
MODELS = [spec for spec in V2['suite']['models'] if spec['name'] in MEMBERS]
assert len(MODELS) == len(MEMBERS), f'selected candidate uses external models: {MEMBERS}'
V2_CSV = v2_metrics_files[0].parent / 'candidates' / SELECTED / 'submission.csv'
assert V2_CSV.exists(), V2_CSV
print('data root:', DATA)
print('v2 selection:', SELECTED, '| validation PQ', round(V2['candidates'][SELECTED]['validation']['pq'], 4))
print('post-processing (from v2 calibration):', POST)
for spec in MODELS:
    print(spec['name'], {k: spec['cfg'][k] for k in ('encoder', 'crop', 'epochs', 'steps_per_epoch', 'batch_size', 'ema_decay', 'seed')})"""

DEBUG = """# Every stage on a handful of images (both GPU workers, test inference, CSV, agreement check): a few minutes.
debug_models = [spec for spec in suite.default_suite(debug=True)['models'] if spec['name'] in MEMBERS]
debug_refit = suite.run_refit(DATA, SPLIT, WORK_ROOT / 'debug_refit', debug_models, POST, debug=True)
print(suite.compare_submissions(V2_CSV, WORK_ROOT / 'debug_refit' / 'submission.csv', DATA / 'test/test_images'))
shutil.rmtree(WORK_ROOT / 'debug_refit' / 'models', ignore_errors=True)
print('debug refit finished: every stage ran (the tiny debug models are untrained).')"""

FULL = """RUN_FULL = torch.cuda.is_available()   # the full refit needs GPUs
if RUN_FULL:
    refit = suite.run_refit(DATA, SPLIT, WORK_ROOT / 'refit', MODELS, POST)
else:
    print('Skipped: no GPU available.')"""

SUMMARY = """if RUN_FULL:
    agreement = suite.compare_submissions(V2_CSV, WORK_ROOT / 'refit' / 'submission.csv', DATA / 'test/test_images')
    (WORK_ROOT / 'refit' / 'agreement_with_v2.json').write_text(json.dumps(agreement, indent=2))
    print('trained on', refit['train_images'], 'labelled images; members', refit['members'], f"({refit['minutes']:.0f} min)")
    print('candidate CSV:', refit['csv'])
    print('agreement with the validated v2 ensemble on the test images:', json.dumps(agreement, indent=1))
    for name, h in refit['histories'].items():
        print(name, 'final epochs:', [round(r['loss'], 4) for r in h['history'][-3:]])"""


def main() -> None:
    split = json.loads((ROOT / "configs/split.json").read_text())
    split_cell = "SPLIT = json.loads(r'''" + json.dumps(split, separators=(",", ":")) + "''')\n" \
        "print({key: len(value) for key, value in SPLIT.items() if isinstance(value, list)}, " \
        "'| refit trains on', len(SPLIT['training']) + len(SPLIT['validation']), 'images')"
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
        nbformat.v4.new_markdown_cell("## 1. Debug refit (a few minutes)"),
        nbformat.v4.new_code_cell(DEBUG),
        nbformat.v4.new_markdown_cell("## 2. Full refit on 707 images (about 1.5 hours on T4 x2)"),
        nbformat.v4.new_code_cell(FULL),
        nbformat.v4.new_code_cell(SUMMARY),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}
    nbformat.validate(notebook)
    out = ROOT / "notebooks/gpu-refit-707.ipynb"
    nbformat.write(notebook, out)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB, {len(cells)} cells)")


if __name__ == "__main__":
    main()
