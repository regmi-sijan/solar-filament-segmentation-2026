"""Image-only contrast segmentation and exact COCO-mask evaluation.

Only the training segmentation polygons supervise calibration. No spines,
chirality labels, external annotations, or test labels are consumed.
"""
import json
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from pycocotools import mask as coco
cv2.setNumThreads(2)


def load_annotations(data):
    a = json.loads((Path(data)/'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json').read_text())
    records = defaultdict(list); annotations = defaultdict(list)
    for x in a['images']: records[x['file_name']].append(x)
    for x in a['annotations']: annotations[x['image_id']].append(x)
    return records, annotations


def ground_truth(records, annotations):
    result = []
    for record in records:
        masks = [coco.merge(coco.frPyObjects(a['segmentation'],record['height'],record['width']))
                 for a in annotations[record['id']]]
        result.append((record['id'], masks))
    return result


def contrast_maps(path):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None: raise ValueError(f'Cannot read {path}')
    x = im.astype(np.float32)/255
    # Infer the bright solar disk from this image, not coordinates or filenames.
    _, disk = cv2.threshold(im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(disk,8)
    if n < 2: raise ValueError(f'No solar disk: {path}')
    mask = (labels == (1+np.argmax(stats[1:,cv2.CC_STAT_AREA]))).astype(np.uint8)
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    mask[:] = 0; cv2.drawContours(mask,contours,-1,1,cv2.FILLED)
    mask = cv2.erode(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)))
    smooth = cv2.GaussianBlur(x,(0,0),1.0)
    # Local brightness deficit at two scales; denominator stabilizes limb contrast.
    maps=[]
    for sigma in (12.,32.,64.):
        background = cv2.GaussianBlur(smooth,(0,0),sigma)
        maps.append(np.clip((background-smooth)/np.maximum(background,.08),0,1))
    return im, mask, maps


def predict_from_maps(mask, maps, config):
    contrast = maps[config['scale']]
    binary = ((contrast > config['threshold']) & (mask>0)).astype(np.uint8)
    radius = config['closing_radius']
    if radius:
        k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1))
        binary = cv2.morphologyEx(binary,cv2.MORPH_CLOSE,k)
    binary &= mask
    count, labels, stats,_ = cv2.connectedComponentsWithStats(binary,8)
    accepted=[]
    for lab in range(1,count):
        if stats[lab,cv2.CC_STAT_AREA] < config['min_area']: continue
        # Bound maximum component size to reject large image artifacts.
        if stats[lab,cv2.CC_STAT_AREA] > config.get('max_area',100000): continue
        elongation = config.get('min_elongation', 1.)
        if elongation > 1:
            x,y,w,h,_ = stats[lab]
            yy,xx=np.nonzero(labels[y:y+h,x:x+w]==lab)
            if len(xx)<3: continue
            eigen=np.linalg.eigvalsh(np.cov(np.stack([xx,yy])))
            ratio=np.sqrt((eigen[1]+1)/(eigen[0]+1))
            if ratio < elongation: continue
        accepted.append(lab)
    return labels,accepted


def encode_instances(labels, accepted):
    return [coco.encode(np.asfortranarray(labels==lab,dtype=np.uint8)) for lab in accepted]


def score_masks(gt, pred):
    ng,npred=len(gt),len(pred)
    if ng and npred:
        iou=coco.iou(pred,gt,[0]*ng).T
    else: iou=np.zeros((ng,npred),dtype=float)
    hit=iou > .5
    tp=int(hit.sum()); fp=int((hit.sum(axis=0)==0).sum()); fn=int((hit.sum(axis=1)==0).sum())
    total=float(iou[hit].sum())
    nonzero=iou[iou>0]
    return {'tp':tp,'fp':fp,'fn':fn,'sum_iou':total,
            'pq':total/(tp+.5*fp+.5*fn) if tp+.5*fp+.5*fn else 0.,
            'pair_iou':nonzero.tolist(),'pair_dice':(2*nonzero/(1+nonzero)).tolist(),
            'gt_overlap_degrees':(iou>0).sum(axis=1).tolist(),
            'pred_overlap_degrees':(iou>0).sum(axis=0).tolist()}


def aggregate(rows):
    sums={k:sum(r[k] for r in rows) for k in ('tp','fp','fn','sum_iou')}
    denom=sums['tp']+.5*sums['fp']+.5*sums['fn']
    sums['pq']=sums['sum_iou']/denom if denom else 0.
    return sums
