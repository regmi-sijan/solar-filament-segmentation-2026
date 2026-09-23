"""Figures derived from held-out validation results, without tuning the model."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'work/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'work/cache'))
import sys,json
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from collections import defaultdict,Counter
from pycocotools import mask as coco
from filament.core import load_annotations,ground_truth,contrast_maps,predict_from_maps

out=ROOT/'report/figures';out.mkdir(parents=True,exist_ok=True)
r=json.loads((ROOT/'results/validation.json').read_text());rows=r['rows']
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,3,figsize=(7.1,2.4))
for ax,key,title in zip(axs[:2],['pair_iou','pair_dice'],['Overlapping-pair IoU','Overlapping-pair Dice']):
 v=[x for row in rows for x in row[key]];ax.hist(v,bins=np.linspace(0,1,21),color='#2878a4',edgecolor='white');ax.set_title(title);ax.set_xlabel('Score');ax.set_ylabel('Pair count')
g=[x for row in rows for x in row['gt_overlap_degrees']];p=[x for row in rows for x in row['pred_overlap_degrees']]
for values,label,shift in [(g,'GT → predictions',-.18),(p,'Prediction → GT',.18)]:
 c=Counter(min(x,5) for x in values);axs[2].bar(np.arange(6)+shift,[c[i] for i in range(6)],width=.36,label=label)
axs[2].set_xticks(range(6),['0','1','2','3','4','5+']);axs[2].set_xlabel('Overlapping counterparts');axs[2].set_title('Fragmentation / merging');axs[2].legend(fontsize=7)
fig.tight_layout();fig.savefig(out/'metrics.pdf');fig.savefig(out/'metrics.png',dpi=160);plt.close(fig)

fig,ax=plt.subplots(figsize=(7.1,1.2));ax.axis('off')
steps=['Grayscale\n2048 × 2048','Infer + erode\nsolar disk','Gaussian\ncontrast map','Threshold +\nclose gaps','Components\n+ area filter','COCO RLE\nCSV']
for i,text in enumerate(steps):
 x=i/6+.005
 ax.add_patch(FancyBboxPatch((x,.22),.14,.58,boxstyle='round,pad=.009',facecolor='#e9f3fa',edgecolor='#2878a4'))
 ax.text(x+.07,.51,text,ha='center',va='center',fontsize=7)
 if i<5:ax.annotate('',xy=(x+.16,.51),xytext=(x+.142,.51),arrowprops=dict(arrowstyle='->',color='#2878a4'))
fig.tight_layout();fig.savefig(out/'pipeline.pdf');plt.close(fig)

by=defaultdict(list)
for row in rows:by[row['file_name']].append(row['pq'])
ranked=sorted(by,key=lambda n:np.mean(by[n]));names=[ranked[int(q*(len(ranked)-1))] for q in [.1,.5,.9]]
data=ROOT/'MAGFiLO_1.0_Kaggle_2026';records,anns=load_annotations(data);config=json.loads((ROOT/'configs/model.json').read_text())
fig,axs=plt.subplots(2,3,figsize=(7.1,4.6))
for j,name in enumerate(names):
 im,disk,maps=contrast_maps(data/'train/train_images'/name);labels,labs=predict_from_maps(disk,maps,config)
 pred=np.isin(labels,labs);g=ground_truth(records[name],anns)[0][1]
 gt=np.any(coco.decode(g),axis=2) if g else np.zeros_like(pred)
 axs[0,j].imshow(im,cmap='gray');axs[0,j].contour(gt,levels=[.5],colors=['#ff526b'],linewidths=.4);axs[0,j].contour(pred,levels=[.5],colors=['#00bddd'],linewidths=.4)
 axs[0,j].set_title(f'{name[:8]} | mean PQ {np.mean(by[name]):.3f}')
 # Crop centered on the largest GT instance for legible morphology.
 areas=[coco.area(x) for x in g];box=coco.toBbox(g[int(np.argmax(areas))]);cx=box[0]+box[2]/2;cy=box[1]+box[3]/2
 width=max(220,int(max(box[2:])*1.5));x0=max(0,int(cx-width/2));y0=max(0,int(cy-width/2));x1=min(2048,x0+width);y1=min(2048,y0+width)
 axs[1,j].imshow(im[y0:y1,x0:x1],cmap='gray');axs[1,j].contour(gt[y0:y1,x0:x1],levels=[.5],colors=['#ff526b'],linewidths=1);axs[1,j].contour(pred[y0:y1,x0:x1],levels=[.5],colors=['#00bddd'],linewidths=1)
 for ax in axs[:,j]:ax.axis('off')
fig.suptitle('Red: one annotator’s ground truth   |   Cyan: prediction',fontsize=11)
fig.tight_layout();fig.savefig(out/'examples.pdf');fig.savefig(out/'examples.png',dpi=150);plt.close(fig)
print('Figures saved')
