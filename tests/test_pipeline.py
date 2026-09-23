import json,unittest
from pathlib import Path
import numpy as np
import torch
from pycocotools import mask as coco
from filament.core import score_masks,aggregate,predict_from_maps,encode_instances
ROOT=Path(__file__).resolve().parents[1]

class PipelineTests(unittest.TestCase):
    def encode(self,a):return coco.encode(np.asfortranarray(a,dtype=np.uint8))
    def test_perfect_and_empty(self):
        a=np.zeros((16,16),np.uint8);a[2:6,3:10]=1;r=self.encode(a)
        self.assertEqual(score_masks([r],[r])['pq'],1.)
        self.assertEqual(score_masks([r],[])['fn'],1)
        self.assertEqual(score_masks([],[r])['fp'],1)
        self.assertEqual(score_masks([],[])['pq'],0.)
    def test_strict_half_iou(self):
        a=np.zeros((16,16),np.uint8);a[2:6,2:6]=1;b=a.copy();b[6:10,2:6]=1
        self.assertEqual(score_masks([self.encode(a)],[self.encode(b)])['tp'],0)
    def test_official_notebook_parity(self):
        # Execute only the pure official PQ/count definitions, not its example cells.
        import ast
        reference=ROOT/'notebooks/self-evaluation-notebook.ipynb'
        if not reference.exists(): self.skipTest('Optional official reference notebook is not bundled')
        n=json.loads(reference.read_text())
        ns={'torch':torch}
        for cell in n['cells']:
            if cell['cell_type']!='code':continue
            tree=ast.parse(''.join(cell['source']))
            for node in tree.body:
                if isinstance(node,ast.FunctionDef) and node.name in ['get_pq_score','fp_count_hit','fn_count_hit']:
                    ns.setdefault('pd',__import__('pandas'))
                    exec(compile(ast.Module(body=[node],type_ignores=[]),'official','exec'),ns)
        rng=np.random.default_rng(1)
        for ng,npred in [(0,0),(0,2),(3,0),(3,4),(2,2)]:
            gs=[self.encode(rng.random((16,16))>.5) for _ in range(ng)]
            ps=[self.encode(rng.random((16,16))>.5) for _ in range(npred)]
            if gs and ps:ps[0]=gs[0]
            mat=coco.iou(ps,gs,[0]*ng).T if ng and npred else np.zeros((ng,npred))
            df=ns['pd'].DataFrame([{'iou_matrix':torch.tensor(mat),'n_gt':ng,'n_pred':npred}])
            self.assertAlmostEqual(score_masks(gs,ps)['pq'],ns['get_pq_score'](df),places=12)
    def test_date_grouping(self):
        s=json.loads((ROOT/'configs/split.json').read_text())
        self.assertFalse({x[:8] for x in s['training']}&{x[:8] for x in s['validation']})
        self.assertTrue(set(s['calibration'])<=set(s['training']))
    def test_components_and_rle(self):
        m=np.ones((32,32),np.uint8);c=np.zeros((32,32),np.float32)
        c[3:8,3:8]=1;c[20:28,20:28]=1
        labs,ids=predict_from_maps(m,[c],dict(scale=0,threshold=.5,closing_radius=0,min_area=10))
        rr=encode_instances(labs,ids);self.assertEqual(len(rr),2)
        self.assertEqual(sum(int(coco.area(r)) for r in rr),89)
    def test_shape_filter_rejects_compact_component(self):
        m=np.ones((64,64),np.uint8);c=np.zeros((64,64),np.float32)
        c[3:13,3:13]=1;c[30:34,25:55]=1
        labels,ids=predict_from_maps(m,[c],dict(scale=0,threshold=.5,closing_radius=0,min_area=10,min_elongation=2))
        self.assertEqual(len(ids),1)
        self.assertEqual(int(coco.area(encode_instances(labels,ids)[0])),120)
    def test_aggregation_is_micro(self):
        self.assertAlmostEqual(aggregate([dict(tp=1,fp=0,fn=0,sum_iou=1),dict(tp=0,fp=9,fn=9,sum_iou=0)])['pq'],.1)
if __name__=='__main__':unittest.main()
