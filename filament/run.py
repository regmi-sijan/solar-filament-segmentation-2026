"""CLI: split, calibrate, evaluate, predict, verify."""
import argparse,csv,hashlib,itertools,json,time
from pathlib import Path
from collections import defaultdict
import numpy as np
from pycocotools import mask as coco
from .core import load_annotations,ground_truth,contrast_maps,predict_from_maps,encode_instances,score_masks,aggregate
ROOT=Path(__file__).resolve().parents[1]


def save(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(x,indent=2))


def make_split(data):
    records,_=load_annotations(data)
    dates=defaultdict(set)
    for name in records: dates[name[:4]].add(name[:8])
    val_dates=set();cal_dates=set()
    for year,ds in sorted(dates.items()):
        ds=sorted(ds,key=lambda d:hashlib.sha256(('2026:'+d).encode()).hexdigest())
        nval=max(1,round(.2*len(ds)));ncal=max(1,round(.15*len(ds)))
        val_dates.update(ds[:nval]);cal_dates.update(ds[nval:nval+ncal])
    result={'seed':2026,'grouping':'capture date, stratified by year',
            'validation':sorted(n for n in records if n[:8] in val_dates),
            'calibration':sorted(n for n in records if n[:8] in cal_dates),
            'training':sorted(n for n in records if n[:8] not in val_dates)}
    assert not set(result['training'])&set(result['validation'])
    save(ROOT/'configs/split.json',result)
    print({k:len(v) for k,v in result.items() if isinstance(v,list)},flush=True)


def calibrate(data, initial_only=False):
    split=json.loads((ROOT/'configs/split.json').read_text());records,anns=load_annotations(data)
    configs=[dict(scale=s,threshold=t,closing_radius=r,min_area=a,max_area=100000,min_elongation=e)
             for s,t,r,a,e in itertools.product([1,2],[.04,.06,.08,.10],[0,2],[300,800],[1.,2.])]
    if initial_only:
        configs=[dict(scale=s,threshold=t,closing_radius=r,min_area=a,max_area=100000)
                 for s,t,r,a in itertools.product([0,1],[.08,.12,.16,.20],[0,2],[100,300])]
    totals=[[] for _ in configs];start=time.time()
    for ix,name in enumerate(split['calibration']):
        _,mask,maps=contrast_maps(data/'train/train_images'/name)
        gt=ground_truth(records[name],anns)
        encoded_cache={}
        for idx,c in enumerate(configs):
            labels,labs=predict_from_maps(mask,maps,c)
            key=(c['scale'],c['threshold'],c['closing_radius'])
            cached=encoded_cache.setdefault(key,{})
            for lab in labs:
                if lab not in cached: cached[lab]=encode_instances(labels,[lab])[0]
            pred=[cached[lab] for lab in labs]
            for _,g in gt:
                score=score_masks(g,pred)
                totals[idx].append({k:score[k] for k in ('tp','fp','fn','sum_iou')})
        if (ix+1)%5==0: print(f'calibrate {ix+1}/{len(split["calibration"])} {time.time()-start:.1f}s',flush=True)
    if initial_only:
        ranked=sorted([{'config':c,**aggregate(t)} for c,t in zip(configs,totals)],key=lambda x:-x['pq'])
        save(ROOT/'results/calibration-initial.json',{'seconds':time.time()-start,'images':len(split['calibration']),'trials':ranked})
        save(ROOT/'configs/model-initial.json',ranked[0]['config'])
        print('INITIAL BEST',ranked[0],flush=True)
        return
    initial=json.loads((ROOT/'results/calibration-initial.json').read_text())
    ranked=sorted(initial['trials']+[{'config':c,**aggregate(t)} for c,t in zip(configs,totals)],key=lambda x:-x['pq'])
    save(ROOT/'results/calibration.json',{'seconds':time.time()-start,'images':len(split['calibration']),'trials':ranked,'initial_seconds':initial['seconds'],'search':'32 initial plus 64 refinement configurations; refinement selected using calibration only'})
    save(ROOT/'configs/model.json',ranked[0]['config']);print('BEST',ranked[0],flush=True)


def evaluate(data):
    split=json.loads((ROOT/'configs/split.json').read_text());records,anns=load_annotations(data)
    config=json.loads((ROOT/'configs/model.json').read_text());rows=[];start=time.time()
    for ix,name in enumerate(split['validation']):
        tick=time.time();_,mask,maps=contrast_maps(data/'train/train_images'/name)
        labels,labs=predict_from_maps(mask,maps,config);pred=encode_instances(labels,labs)
        elapsed=time.time()-tick
        for iid,g in ground_truth(records[name],anns):
            rows.append({'file_name':name,'image_record':iid,'inference_seconds':elapsed,**score_masks(g,pred)})
        if (ix+1)%20==0: print(f'evaluate {ix+1}/{len(split["validation"])}',flush=True)
    # Cluster bootstrap by capture date preserves related observations and annotators.
    groups=defaultdict(list)
    for r in rows: groups[r['file_name'][:8]].append(r)
    clusters=[aggregate(g) for g in groups.values()];rng=np.random.default_rng(2026)
    samples=[aggregate([clusters[i] for i in rng.integers(0,len(clusters),len(clusters))])['pq'] for _ in range(2000)]
    summary={**aggregate(rows),'pq_bootstrap_95ci':np.percentile(samples,[2.5,97.5]).tolist(),
             'images':len(split['validation']),'annotation_sets':len(rows),'date_clusters':len(groups),
             'total_seconds':time.time()-start,'bootstrap_seed':2026,'config':config}
    save(ROOT/'results/validation.json',{'summary':summary,'rows':rows});print(summary,flush=True)


def predict(data):
    config=json.loads((ROOT/'configs/model.json').read_text());out=ROOT/'results/submission.csv';start=time.time();manifest=[]
    with out.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['filament_id','segmentation_rle'])
        paths=sorted((data/'test/test_images').glob('*.jpeg'))
        for ix,path in enumerate(paths):
            tick=time.time();_,mask,maps=contrast_maps(path)
            labels,labs=predict_from_maps(mask,maps,config);pred=encode_instances(labels,labs)
            for j,rle in enumerate(pred,1): writer.writerow([f'{path.stem}_{j}',rle['counts'].decode('ascii')])
            manifest.append({'file_name':path.name,'instances':len(pred),'seconds':time.time()-tick})
            if (ix+1)%20==0:print(f'predict {ix+1}/{len(paths)}',flush=True)
    save(ROOT/'results/inference-manifest.json',{'config':config,'seconds':time.time()-start,'images':manifest})
    verify(data)


def verify(data):
    ids=set();names={p.stem for p in (data/'test/test_images').glob('*.jpeg')};seen=set();count=0
    with (ROOT/'results/submission.csv').open() as f:
        reader=csv.DictReader(f);assert reader.fieldnames==['filament_id','segmentation_rle']
        for row in reader:
            fid=row['filament_id'];assert fid not in ids;ids.add(fid)
            name,index=fid.rsplit('_',1);assert name in names and int(index)>0;seen.add(name)
            rle={'size':[2048,2048],'counts':row['segmentation_rle'].encode('ascii')}
            mask=coco.decode(rle);assert mask.shape==(2048,2048) and mask.any()
            assert coco.encode(np.asfortranarray(mask))['counts']==rle['counts'];count+=1
    manifest=json.loads((ROOT/'results/inference-manifest.json').read_text())
    assert {Path(x['file_name']).stem for x in manifest['images']}==names
    assert sum(x['instances'] for x in manifest['images'])==count
    result={'status':'passed','rows':count,'test_images_processed':len(names),'images_with_predictions':len(seen),
            'images_without_predictions':sorted(names-seen),'sha256':hashlib.sha256((ROOT/'results/submission.csv').read_bytes()).hexdigest()}
    save(ROOT/'results/submission-verification.json',result);print(result,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['split','calibrate-initial','calibrate','evaluate','predict','verify']);p.add_argument('--data',type=Path,default=ROOT/'MAGFiLO_1.0_Kaggle_2026');a=p.parse_args()
    {'split':make_split,'calibrate-initial':lambda d:calibrate(d,initial_only=True),'calibrate':calibrate,'evaluate':evaluate,'predict':predict,'verify':verify}[a.stage](a.data)
if __name__=='__main__':main()
