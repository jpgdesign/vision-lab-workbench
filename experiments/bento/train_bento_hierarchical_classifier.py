"""Train an optional coarse-conditioned fine classifier on frozen YOLO26 food features.

Only a linear head is optimized. Fine GT is source-defined exploratory supervision,
not a claim of complete manually verified hierarchy. Coarse detections are never
modified. Deployment acceptance remains disabled until predicted-crop validation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image

from bento_research.common import digest, now, save


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--coarse-weights',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--imgsz',type=int,default=224)
    parser.add_argument('--batch',type=int,default=16)
    parser.add_argument('--epochs',type=int,default=80)
    parser.add_argument('--patience',type=int,default=12)
    parser.add_argument('--views',type=int,default=2)
    parser.add_argument('--lr',type=float,default=.003)
    parser.add_argument('--weight-decay',type=float,default=.01)
    args=parser.parse_args()
    if args.imgsz<64 or args.imgsz%32 or min(args.batch,args.epochs,args.patience,args.views)<1:
        raise ValueError('Require positive batch/epochs/patience/views and stride32 image size')
    args.dataset=args.dataset.resolve(); args.coarse_weights=args.coarse_weights.resolve(); args.output=args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()): raise FileExistsError('Use a new output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    state={'status':'running','stage':'auditing','started_at':now(),'pid':os.getpid(),'epochs':args.epochs}
    def progress(**values):
        state.update(values); state['updated_at']=now(); save(args.output/'progress.json',state)
        print(json.dumps(values,ensure_ascii=True),flush=True)
    progress()
    try:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        import torch
        from torch import nn
        from safetensors.torch import save_file
        from bento_research.hierarchical_refiner import FrozenCoarseFeatures, crop_preprocess, state_digest, parent_mask, masked_logits
        torch.manual_seed(42); random.seed(42); np.random.seed(42); torch.set_num_threads(4)
        torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        dataset=read(args.dataset); classes=dataset['classes']
        samples=dataset.get('samples') or [row for values in dataset['splits'].values() for row in values]
        if [row['id'] for row in classes] != list(range(len(classes))): raise ValueError('Class IDs must be contiguous')
        if len({row['name'] for row in classes}) != len(classes): raise ValueError('Duplicate class names')
        mask=parent_mask(classes)
        supervised={split:[row for row in samples if row['split']==split and row.get('supervised_fine') and row.get('class_id') is not None] for split in ['train','val','test']}
        if not supervised['train'] or not supervised['val']: raise ValueError('Supervised train and independent validation crops required')
        for split,rows in supervised.items():
            for row in rows:
                if not mask[int(row['parent_id']),int(row['class_id'])]: raise ValueError('Fine class/parent conflict')
                if not Path(row['crop_path']).is_file(): raise FileNotFoundError(row['crop_path'])
        support={row['id']:len({sample['rgb_sha256'] for sample in supervised['train'] if sample['class_id']==row['id']}) for row in classes}
        if any(count<5 for count in support.values()): raise ValueError('Each enabled fine class requires at least5 distinct train images')
        split_hashes={split:{row['rgb_sha256'] for row in samples if row['split']==split} for split in supervised}
        for a,b in [('train','val'),('train','test'),('val','test')]:
            if split_hashes[a]&split_hashes[b]: raise ValueError(f'Image leakage {a}/{b}')
        dataset_sha=digest(args.dataset); coarse_sha=digest(args.coarse_weights)
        extractor=FrozenCoarseFeatures(args.coarse_weights,args.device)
        frozen_before=state_digest(extractor.native)
        progress(stage='extracting_frozen_neck_features',supervised_counts={key:len(value) for key,value in supervised.items()},fine_classes=len(classes))
        embeddings={}; embedding_audit={}; feature_dim=None
        for split,rows in supervised.items():
            tasks=[(row,view) for row in rows for view in range(args.views if split=='train' else 1)]
            chunks=[]; labels=[]; parents=[]; record_ids=[]; crop_hashes=[]
            for offset in range(0,len(tasks),args.batch):
                batch=tasks[offset:offset+args.batch]; images=[]
                for row,view in batch:
                    crop_digest=digest(Path(row['crop_path']))
                    if row.get('crop_sha256') != crop_digest:
                        raise RuntimeError(f"Crop differs from the audited dataset: {row['crop_path']}")
                    with Image.open(row['crop_path']) as image:
                        stable=int(str(row.get('id',row.get('crop_id'))).encode().hex()[-8:],16)
                        tensor=crop_preprocess(image,args.imgsz,augmentation_seed=42+stable+view*1000003 if view else None)
                    images.append(tensor); labels.append(int(row['class_id'])); parents.append(int(row['parent_id']))
                    record_ids.append(str(row.get('id',row.get('crop_id')))+f':view{view}')
                    crop_hashes.append(crop_digest)
                with torch.inference_mode(): features=extractor(torch.stack(images).to(args.device)).cpu().numpy()
                if not np.isfinite(features).all(): raise RuntimeError('Frozen features contain NaN/Inf')
                chunks.append(features)
                progress(stage='extracting_frozen_neck_features',extracting_split=split,extracted=min(offset+len(batch),len(tasks)),extract_total=len(tasks))
            array=np.concatenate(chunks) if chunks else np.empty((0,feature_dim or 0),dtype=np.float32)
            if feature_dim is None: feature_dim=array.shape[1]
            embeddings[split]={'x':array,'y':np.array(labels,dtype=np.int64),'parent':np.array(parents,dtype=np.int64)}
            np.savez_compressed(args.output/f'features-{split}.npz',**embeddings[split])
            embedding_audit[split]={'record_ids':record_ids,'crop_sha256':crop_hashes,'samples':len(rows),'views':args.views if split=='train' else 1}
        frozen_after=state_digest(extractor.native)
        if frozen_before != frozen_after or coarse_sha != digest(args.coarse_weights): raise RuntimeError('Frozen detector changed during extraction')
        save(args.output/'feature_audit.json',{'dataset_sha256':dataset_sha,'coarse_checkpoint_sha256':coarse_sha,'detector_state_before':frozen_before,
            'detector_state_after':frozen_after,'feature_layers':extractor.feature_indices,'splits':embedding_audit})
        # Features are copied to CPU; the optimizer never receives a detector tensor.
        del extractor
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        train=embeddings['train']; mean=train['x'].mean(0); std=np.maximum(train['x'].std(0),1e-3)
        np.savez_compressed(args.output/'scaler.npz',mean=mean,std=std)
        normalized={split:{'x':torch.from_numpy((values['x']-mean)/std),'y':torch.from_numpy(values['y']),
                           'parent':torch.from_numpy(values['parent'])} for split,values in embeddings.items()}
        model=nn.Linear(feature_dim,len(classes))
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        counts=np.bincount(train['y'],minlength=len(classes)); class_weights=np.sqrt(counts.mean()/np.maximum(counts,1))
        class_weights=np.minimum(class_weights,3.).astype(np.float32)
        criterion=nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights))
        config={'schema_version':1,'created_at':now(),'scope':'Source-defined fine crop classifier; not complete manually validated hierarchy',
                'classes':classes,'coarse_checkpoint_sha256':coarse_sha,'coarse_checkpoint_filename':args.coarse_weights.name,
                'dataset_sha256':dataset_sha,'feature_dimension':feature_dim,'feature_extractor':'Frozen successful six-class YOLO neck P3/P4/P5, pooled concatenation',
                'imgsz':args.imgsz,'head':'linear','optimizer':'AdamW','lr':args.lr,'weight_decay':args.weight_decay,
                'max_epochs':args.epochs,'early_stopping_patience':args.patience,'seed':42,'train_views':args.views,
                'class_weights':class_weights.tolist(),'minimum_distinct_train_images':5,'train_unique_support':support,
                'preservation':'Detector parameters never enter optimizer; coarse outputs independent of classifier',
                'calibration':'GT-crop diagnostic only here; detector-predicted validation crops required before deployment'}
        save(args.output/'config.json',config)
        best=float('inf'); best_epoch=0; history=[]; started=time.monotonic()
        def measure(split):
            values=normalized[split]
            if not len(values['y']): return {'samples':0,'loss':None,'accuracy':None},np.empty((0,len(classes)),dtype=np.float32)
            model.eval()
            with torch.no_grad():
                logits=masked_logits(model,values['x'],values['parent'],mask)
                loss=float(torch.nn.functional.cross_entropy(logits,values['y']))
                probabilities=logits.softmax(1)
                accuracy=float((probabilities.argmax(1)==values['y']).float().mean())
            return {'samples':len(values['y']),'loss':loss,'accuracy':accuracy},probabilities.numpy()
        for epoch in range(1,args.epochs+1):
            model.train(); permutation=torch.randperm(len(normalized['train']['y'])); total=0.; steps=0
            for indices in permutation.split(64):
                optimizer.zero_grad(set_to_none=True)
                values=normalized['train']; logits=masked_logits(model,values['x'][indices],values['parent'][indices],mask)
                loss=criterion(logits,values['y'][indices])
                if not torch.isfinite(loss): raise RuntimeError('Non-finite classifier loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); optimizer.step()
                total+=float(loss.detach()); steps+=1
            validation,_=measure('val'); improved=validation['loss']<best-1e-7
            if improved:
                best=validation['loss']; best_epoch=epoch
                save_file({key:value.detach().cpu().contiguous() for key,value in model.state_dict().items()},str(args.output/'head.safetensors'))
            record={'epoch':epoch,'train_weighted_loss':total/steps,'validation':validation,'best_epoch':best_epoch}
            history.append(record); save(args.output/'history.json',history)
            progress(stage='training_linear_head',completed_epochs=epoch,latest=record,seconds=time.monotonic()-started)
            if epoch-best_epoch>=args.patience: break
        from safetensors.torch import load_file
        model.load_state_dict(load_file(str(args.output/'head.safetensors')),strict=True)
        results={}; predictions={}
        for split in ['train','val','test']:
            result,probabilities=measure(split); results[split]=result
            rows=supervised[split]
            if split=='train': rows=[row for row in rows for _ in range(args.views)]
            records=[]
            for row,scores in zip(rows,probabilities):
                candidate=int(scores.argmax()); records.append({'sample_id':row.get('id',row.get('crop_id')),'source_id':row['source_id'],
                    'parent_id':int(row['parent_id']),'true_class_id':int(row['class_id']),'candidate_class_id':candidate,
                    'candidate_class_name':classes[candidate]['name'],'confidence':float(scores[candidate]),'correct':candidate==row['class_id']})
            predictions[split]=records; save(args.output/f'gt-crop-predictions-{split}.json',records)
        gt_thresholds={'schema_version':1,'scope':'diagnostic_GT_validation_crops_only_not_deployment','parents':{}}
        for parent in range(6):
            rows=[row for row in predictions['val'] if row['parent_id']==parent]
            options=[]
            for threshold in sorted({row['confidence'] for row in rows}):
                selected=[row for row in rows if row['confidence']>=threshold]
                if len(selected)>=5 and sum(row['correct'] for row in selected)/len(selected)>=.9:
                    options.append((len(selected),threshold,sum(row['correct'] for row in selected)/len(selected)))
            chosen=max(options,key=lambda value:(value[0],-value[1])) if options and int(mask[parent].sum())>=2 else None
            gt_thresholds['parents'][str(parent)]={'enabled':bool(chosen),'threshold':chosen[1] if chosen else None,
                'accepted':chosen[0] if chosen else 0,'precision':chosen[2] if chosen else None,'known_validation_samples':len(rows),
                'reason':'diagnostic_only_supported_GT_crops' if chosen else 'insufficient_validation_support_or_precision'}
        save(args.output/'gt_crop_thresholds.json',gt_thresholds)
        save(args.output/'thresholds.json',{'schema_version':1,'scope':'detector_val_crops_not_yet_calibrated',
            'parents':{str(parent):{'enabled':False,'threshold':None,'reason':'no_detector_validation_calibration'} for parent in range(6)}})
        if digest(args.coarse_weights)!=coarse_sha or digest(args.dataset)!=dataset_sha: raise RuntimeError('Source checkpoint or dataset changed during head training')
        evaluation={'status':'completed','best_epoch':best_epoch,'completed_epochs':len(history),'gt_crop_metrics_supported_classes_only':results,
                    'coarse_detector_file_unchanged':True,'detector_state_unchanged_during_feature_extraction':frozen_before==frozen_after,
                    'coarse_checkpoint_sha256':coarse_sha,'fine_head_sha256':digest(args.output/'head.safetensors'),
                    'deployment_enabled':False,'limitations':['GT crops are easier than predicted crops; metrics are not end-to-end detector performance',
                        'Only supported source-defined fine labels are optimized; no labels are invented for unknown/unsupported crops',
                        'Test crops never participate in feature normalization, optimizer, early stopping or threshold selection',
                        'Detector-predicted validation calibration and end-to-end retention evaluation remain separate']}
        save(args.output/'evaluation.json',evaluation)
        progress(status='completed',stage='completed',best_epoch=best_epoch,completed_at=now(),metrics=results)
    except BaseException as exc:
        progress(status='failed',error=f'{type(exc).__name__}: {exc}',failed_at=now())
        raise


if __name__=='__main__': main()
