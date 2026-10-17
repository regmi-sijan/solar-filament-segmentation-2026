"""Build notebooks/gpu-treesel.ipynb: tree-based candidate selection on the saved v2 / refit models (Kaggle T4 x2).

Inputs on Kaggle: the competition data, the output of the v2 suite notebook (weights, suite_metrics.json) and the
output of the refit notebook (last.pt of the 707-image models).

    python scripts/create_gpu_notebook_treesel.py
"""
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]

INTRO = """# Solar filament segmentation: tree-based candidate selection

The fixed post-processing rule (hysteresis thresholds + grouping) turns a probability map into instances. This notebook
tests a different last step on the **same** probability maps: build a tree of connected components over a ladder of
thresholds, train a scorer on the calibration images to predict each component's expected matched IoU, and pick a set
of disjoint components that maximises the expected micro PQ (exact dynamic programme, one global cut-off tau).

Protocol: the models are the v2 models trained on 457 images (no calibration or validation images). The scorer and tau are
fitted on the 111 calibration images only (scorer mode and tau by cross-fitting over capture dates). The 139 validation
images are scored once and compared with the v2 fixed rule on the very same maps (v2 validation PQ 0.4271).
The final scorer (calibration + validation) writes test CSVs for the held-out models and for the 707-image refit models.

**Nothing is submitted to Kaggle by this notebook.**"""

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
from filament import suite, treesel
from filament.dl import PostConfig, list_images
print('modules:', suite.__file__, treesel.__file__)"""

LOCATE = """annotation_files = list(INPUT_ROOT.rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
assert len(annotation_files) == 1, f'Attach the competition data; found {len(annotation_files)} annotation files'
DATA = annotation_files[0].parent.parent
v2_files = sorted(INPUT_ROOT.rglob('v2/suite_metrics.json'))
assert v2_files, 'Attach the output of the v2 suite notebook'
V2_DIR = v2_files[0].parent
V2 = json.loads(v2_files[0].read_text())
SELECTED = V2['selected_by_calibration']
MEMBERS = V2['candidates'][SELECTED]['members']
POST = V2['candidates'][SELECTED]['post']
MODELS = [spec for spec in V2['suite']['models'] if spec['name'] in MEMBERS]
assert len(MODELS) == len(MEMBERS)
held_out = [{'name': s['name'], 'encoder': s['cfg']['encoder'], 'cost': s.get('cost', 1.0),
             'weights': str(V2_DIR / 'models' / s['name'] / 'best.pt')} for s in MODELS]
refit_files = sorted(INPUT_ROOT.rglob('refit/refit_metrics.json'))
refit = []
if refit_files:
    refit = [{'name': s['name'], 'encoder': s['cfg']['encoder'], 'cost': s.get('cost', 1.0),
              'weights': str(refit_files[0].parent / 'models' / s['name'] / 'last.pt')} for s in MODELS]
for spec in held_out + refit:
    assert Path(spec['weights']).exists(), spec['weights']
print('data root:', DATA, '| members:', MEMBERS, '| refit models found:', bool(refit))
print('fixed-rule post-processing (v2, calibration-tuned):', POST)"""

INFER = """# probability maps (8-bit PNG, flip TTA) for calibration + validation + test with the held-out models,
# and for test only with the 707-image refit models
train_dir = DATA / 'train/train_images'
test_paths = list_images(DATA / 'test/test_images')
label_names = list(SPLIT['calibration']) + list(SPLIT['validation'])
held_images = {Path(n).stem: str(train_dir / n) for n in label_names}
held_images.update({p.stem: str(p) for p in test_paths})
PROBS = WORK_ROOT / 'probs'
work = WORK_ROOT / 'treesel_work'
work.mkdir(exist_ok=True)
t0 = time.time()
suite._wait(suite._launch('infer', held_out, {'probs': str(PROBS / 'held_out'), 'tta': True, 'images': held_images}, work))
if refit:
    suite._wait(suite._launch('infer', refit, {'probs': str(PROBS / 'refit'), 'tta': True,
                                               'images': {p.stem: str(p) for p in test_paths}}, work))
print(f'inference finished in {(time.time() - t0) / 60:.1f} min')"""

DEBUG = """# every stage on a handful of images (uses the maps above)
dbg = treesel.run_treesel(DATA, SPLIT, PROBS / 'held_out', MEMBERS, WORK_ROOT / 'treesel_debug',
                          baseline_post=PostConfig(**POST), debug=True)
print({k: dbg[k] for k in ('mode', 'tau', 'csv')})"""

FULL = """res = treesel.run_treesel(DATA, SPLIT, PROBS / 'held_out', MEMBERS, WORK_ROOT / 'treesel',
                          baseline_post=PostConfig(**POST),
                          refit=({'probs_dir': str(PROBS / 'refit'), 'members': MEMBERS} if refit else None))"""

SUMMARY = """v = res['validation']
print(f"scorer mode {res['mode']}, tau {res['tau']:.3f}")
print(f"VALIDATION tree selection : PQ {v['pq']:.4f}  CI {v['pq_bootstrap_95ci']}  TP/FP/FN {v['tp']}/{v['fp']}/{v['fn']}")
b = res['fixed_rule_baseline_validation']
print(f"VALIDATION fixed rule     : PQ {b['pq']:.4f}  TP/FP/FN {b['tp']}/{b['fp']}/{b['fn']}   (v2 reported 0.4271)")
print('oracle ceilings (true expected IoU as score):', res['oracle'])
for m in ('hurdle', 'direct'):
    print(m, 'calibration out-of-fold PQ', round(res['calibration_tuning'][m]['best_pq'], 4), 'tau', res['calibration_tuning'][m]['best_tau'])
print('validation PQ by tau (reported, not used to choose):', {k: round(x, 4) for k, x in res['validation_tau_sensitivity'].items()})
print('test CSVs:', json.dumps(res['csv'], indent=1))"""


def main() -> None:
    split = json.loads((ROOT / "configs/split.json").read_text())
    split_cell = "SPLIT = json.loads(r'''" + json.dumps(split, separators=(",", ":")) + "''')\n" \
        "print({key: len(value) for key, value in SPLIT.items() if isinstance(value, list)})"
    files = [("code/filament/__init__.py", '"""Solar filament segmentation package (written by the notebook)."""\n')]
    files += [(f"code/filament/{n}.py", (ROOT / f"filament/{n}.py").read_text()) for n in ("core", "dl", "suite", "treesel")]
    cells = [nbformat.v4.new_markdown_cell(INTRO), nbformat.v4.new_code_cell(SETUP)]
    cells += [nbformat.v4.new_code_cell(f"%%writefile {path}\n{source.rstrip()}\n") for path, source in files]
    cells += [
        nbformat.v4.new_code_cell(IMPORT),
        nbformat.v4.new_code_cell(split_cell),
        nbformat.v4.new_code_cell(LOCATE),
        nbformat.v4.new_markdown_cell("## 1. Probability maps from the saved models"),
        nbformat.v4.new_code_cell(INFER),
        nbformat.v4.new_markdown_cell("## 2. Debug pass (minutes)"),
        nbformat.v4.new_code_cell(DEBUG),
        nbformat.v4.new_markdown_cell("## 3. Full run: tune on calibration, score validation once, write test CSVs"),
        nbformat.v4.new_code_cell(FULL),
        nbformat.v4.new_code_cell(SUMMARY),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}
    nbformat.validate(notebook)
    out = ROOT / "notebooks/gpu-treesel.ipynb"
    nbformat.write(notebook, out)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB, {len(cells)} cells)")


if __name__ == "__main__":
    main()
