"""Read-only dataset audit; writes aggregate results to docs/."""
import json, hashlib
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'MAGFiLO_1.0_Kaggle_2026'
a = json.loads((DATA / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json').read_text())
images = a['images']; anns = a['annotations']
by_name = Counter(i['file_name'] for i in images)
image_ids = {i['id'] for i in images}
counts = Counter(v['image_id'] for v in anns)
modes, sizes, hashes = Counter(), Counter(), defaultdict(list)
errors = []
for split in ['train', 'test']:
    for path in sorted((DATA / split / f'{split}_images').glob('*.jpeg')):
        try:
            with Image.open(path) as im:
                im.load(); modes[im.mode] += 1; sizes[str(im.size)] += 1
            hashes[hashlib.sha256(path.read_bytes()).hexdigest()].append(f'{split}/{path.name}')
        except Exception as e:
            errors.append({'file':str(path), 'error':str(e)})
areas = np.asarray([v['area'] for v in anns], dtype=float)
invalid_polygons = []
for v in anns:
    seg = v['segmentation']
    if not isinstance(seg,list) or not seg or any(len(poly)<6 or len(poly)%2 for poly in seg):
        invalid_polygons.append(v['id'])
train_names = {p.name for p in (DATA/'train/train_images').glob('*.jpeg')}
test_names = {p.name for p in (DATA/'test/test_images').glob('*.jpeg')}
r = {
 'training_images':len(train_names),'test_images':len(test_names),
 'annotation_records':len(anns),'annotation_image_records':len(images),
 'annotation_sets_per_source_image':dict(sorted(Counter(by_name.values()).items())),
 'image_modes':dict(modes),'image_dimensions':dict(sizes),'unreadable_images':errors,
 'duplicate_image_record_ids':len(images)-len(image_ids),
 'duplicate_annotation_ids':len(anns)-len({v['id'] for v in anns}),
 'orphan_annotations':sum(v['image_id'] not in image_ids for v in anns),
 'image_records_without_annotations':sum(counts[i['id']]==0 for i in images),
 'invalid_polygon_structures':invalid_polygons,
 'nonpositive_annotation_areas':int((areas<=0).sum()),
 'area_pixels_percentiles':dict(zip(['min','p25','median','p75','p95','max'],np.percentile(areas,[0,25,50,75,95,100]).tolist())),
 'train_test_filename_overlap':sorted(train_names & test_names),
 'byte_identical_image_groups':[v for v in hashes.values() if len(v)>1],
 'annotations_per_image_record_percentiles':dict(zip(['min','median','p95','max'],np.percentile([counts[i['id']] for i in images],[0,50,95,100]).tolist())),
 'training_years':dict(sorted(Counter(n[:4] for n in train_names).items())),
 'test_years':dict(sorted(Counter(n[:4] for n in test_names).items()))}
(ROOT/'docs/data-audit.json').write_text(json.dumps(r,indent=2))
print(json.dumps(r,indent=2))
