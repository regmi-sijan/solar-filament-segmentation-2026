"""Assemble an explicit source release without raw data, environments, or credentials."""
from pathlib import Path
import shutil,zipfile,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'release';DEST.mkdir(exist_ok=True)
allowed=['README.md','LICENSE','THIRD_PARTY_NOTICES.md','.gitignore','requirements.txt','requirements-lock.txt']
for folder,patterns in {'filament':['*.py'],'configs':['*.json'],'tests':['*.py'],'scripts':['*.py'],
                       'docs':['method-walkthrough.md','requirements-checklist.md','data-audit.json','dataset-verification.json','submission-status.json'],
                       'notebooks':['competition-pipeline.ipynb','kaggle-inference.ipynb'],
                       'results':['calibration.json','calibration-initial.json','validation.json','submission.csv','submission-verification.json','inference-manifest.json'],
                       'report':['main.tex','preamble.tex','main.bib','main.pdf'],'report/figures':['*.pdf','*.png']}.items():
    for pattern in patterns:
        allowed.extend(str(p.relative_to(ROOT)) for p in sorted((ROOT/folder).glob(pattern)) if p.is_file())
manifest=[]
for name in sorted(set(allowed)):
    src=ROOT/name
    if not src.exists():raise FileNotFoundError(src)
    dst=DEST/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    manifest.append({'path':name,'bytes':src.stat().st_size,'sha256':hashlib.sha256(src.read_bytes()).hexdigest()})
(DEST/'release-manifest.json').write_text(json.dumps(manifest,indent=2))
with zipfile.ZipFile(ROOT/'competition-package.zip','w',zipfile.ZIP_DEFLATED) as z:
    for row in manifest:z.write(DEST/row['path'],'solar-filament-segmentation-2026/'+row['path'])
    z.write(DEST/'release-manifest.json','solar-filament-segmentation-2026/release-manifest.json')
print('Release files:',len(manifest),'ZIP bytes:',(ROOT/'competition-package.zip').stat().st_size)
