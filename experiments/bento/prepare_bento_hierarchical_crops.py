"""Prepare conservative source-labeled fine crops without changing frozen coarse data.

This is NOT a completed annotation-v2 benchmark. Source names provide limited
supervision; unknown, ambiguous and unsupported labels never become negatives.
Original six-class validation/test images and related groups stay out of training.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path

from PIL import Image, ImageOps

BLOCK_FLAGS = {'semantic_review_required', 'source_box_definition_still_needs_alignment',
               'overlaps_more_specific_source_classes', 'incomplete_fine_supervision',
               'possible_parent_child_overlap'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''): value.update(chunk)
    return value.hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def group_keys(row):
    return {value for value in [row.get('group_id'), row.get('old_group_id')] if value}


def crop_bounds(box, width, height):
    x, y, w, h = map(float, box)
    if not all(math.isfinite(v) for v in (x,y,w,h)) or w <= 0 or h <= 0:
        raise ValueError('Invalid normalized box')
    if x < -1e-6 or y < -1e-6 or x+w > 1+1e-6 or y+h > 1+1e-6:
        raise ValueError('Source crop exceeds image boundaries')
    bounds = [max(0, math.floor(x*width)), max(0, math.floor(y*height)),
              min(width, math.ceil((x+w)*width)), min(height, math.ceil((y+h)*height))]
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError('Empty crop')
    return bounds


def build(args):
    coarse_dir = args.coarse_dataset.resolve()
    full_dir = args.full_dataset.resolve()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    input_files = [coarse_dir/'manifest.json', coarse_dir/'class_mapping.json', coarse_dir/'legacy_holdout.json',
                   full_dir/'manifest.json', full_dir/'class_mapping.json', full_dir/'validation_checks.json']
    initial_hashes = {str(path): sha(path) for path in input_files}
    coarse, full = read(coarse_dir/'manifest.json'), read(full_dir/'manifest.json')
    coarse_mapping, full_mapping = read(coarse_dir/'class_mapping.json'), read(full_dir/'class_mapping.json')
    source_mapping = {(row['source'],row['source_class_id']):row for row in coarse_mapping['mapping']}
    canonical = {row['id']:row for row in full_mapping['canonical_classes']}
    parent_candidates = defaultdict(set)
    for row in full_mapping['mapping']:
        if row.get('canonical_class_id') is None: continue
        parent_candidates[row['canonical_class_id']].add(source_mapping[(row['source'],row['source_class_id'])]['target_class_id'])
    assert len(parent_candidates)==149 and all(len(values)==1 for values in parent_candidates.values())
    parents = {key:next(iter(values)) for key,values in parent_candidates.items()}
    by_source = {row['source_id']:row for row in full['images']}
    fixed = {split:[by_source[row['source_id']] for row in coarse['images'] if row['split']==split]
             for split in ('val','test')}
    assert len(fixed['val'])==len(fixed['test'])==59
    coarse_by_source = {row['source_id']:row for row in coarse['images']}
    for split, rows in fixed.items():
        for row in rows:
            old=coarse_by_source[row['source_id']]
            assert old['split']==split and old['rgb_sha256']==row['rgb_sha256']
            assert old['original_sha256']==row['sha256']
            assert len(old['objects'])==len(row['objects'])
            candidates=list(row['objects'])
            for obj in old['objects']:
                matches=[i for i,other in enumerate(candidates)
                         if other['original_class']==obj['original_class']
                         and parents[other['class_id']]==obj['class_id']
                         and max(abs(a-b) for a,b in zip(other['bbox_xywh_normalized'],obj['bbox_xywh_normalized']))<1e-6]
                assert len(matches)==1; candidates.pop(matches[0])
    protected = fixed['val']+fixed['test']+full['legacy_samples']
    protected_ids = {row['source_id'] for row in protected}
    protected_rgb = {row['rgb_sha256'] for row in protected}
    protected_groups = set().union(*(group_keys(row) for row in protected))
    protected_groups |= set(full['grouping']['legacy_holdout_group_ids'])
    protected_groups |= set(coarse['grouping']['legacy_holdout_group_ids'])
    protected_groups |= {row['group_id'] for row in coarse['images'] if row['split'] in ('val','test')}
    train_rows=[]; excluded_images=[]
    for row in full['images']:
        if row['split']!='train': continue
        reasons=[]
        if set(row.get('source_ids',[row['source_id']]))&protected_ids: reasons.append('protected_source_id')
        if row['rgb_sha256'] in protected_rgb: reasons.append('protected_rgb')
        if group_keys(row)&protected_groups: reasons.append('protected_related_group')
        if reasons: excluded_images.append({'source_id':row['source_id'],'reasons':reasons, 'groups':sorted(group_keys(row))})
        else: train_rows.append(row)
    split_rows={'train':train_rows, **fixed}
    split_groups={split:set().union(*(group_keys(row) for row in rows)) for split,rows in split_rows.items()}
    split_rgbs={split:{row['rgb_sha256'] for row in rows} for split,rows in split_rows.items()}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert not split_groups[a]&split_groups[b], f'Group overlap {a}/{b}'
        assert not split_rgbs[a]&split_rgbs[b], f'RGB overlap {a}/{b}'
    assert not split_groups['train']&protected_groups
    original_train_groups={row['group_id'] for row in coarse['images'] if row['split']=='train'}
    assert not original_train_groups & (split_groups['val']|split_groups['test'])
    sources_before={}; labels_before={}; samples=[]; image_records={split:[] for split in split_rows}
    supported_rgbs=defaultdict(set); supported_groups=defaultdict(set)
    for split,rows in split_rows.items():
        for row in sorted(rows,key=lambda item:item['source_id']):
            image_path=(full_dir/row['image']).resolve(); label_path=(full_dir/row['label']).resolve()
            image_record={key:row[key] for key in ['source_id','source','source_path','rgb_sha256','group_id','old_group_id','width','height']}
            image_record.update(split=split, image_path=str(image_path), image_sha256=row['sha256'],
                                original_dataset_split=row['split'], objects=[])
            for index,obj in enumerate(row['objects']):
                fine_id=obj['class_id']; cat=canonical[fine_id]; parent=parents[fine_id]
                reasons=[]
                flags=set(cat['flags'])|set(obj.get('flags',[]))
                if obj['kind']!='named_ingredient_or_dish': reasons.append('not_named_fine_source_label')
                reasons += sorted(flags&BLOCK_FLAGS)
                if parent is None: reasons.append('no_unambiguous_coarse_parent')
                bounds=crop_bounds(obj['bbox_xywh_normalized'],row['width'],row['height'])
                usable=min(bounds[2]-bounds[0],bounds[3]-bounds[1])>=args.min_crop_side
                if not usable: reasons.append('crop_too_small')
                known=not reasons
                sample={
                    'id':f"s{row['source_id']:04d}_o{index:03d}", 'crop_id':f"s{row['source_id']:04d}_o{index:03d}",
                    'split':split, 'image_path':str(image_path), 'image_sha256':row['sha256'],
                    'source_id':row['source_id'], 'source':row['source'], 'source_path':row['source_path'],
                    'rgb_sha256':row['rgb_sha256'], 'group_id':row['group_id'], 'old_group_id':row['old_group_id'],
                    'source_object_index':index, 'original_class_id':obj['original_class_id'],
                    'original_class':obj['original_class'], 'original_annotation_type':obj['original_type'],
                    'canonical_class_id':fine_id, 'class_name':obj['class_name'], 'class_id':None,
                    'parent_id':parent, 'parent_name':coarse_mapping['classes'][parent] if parent is not None else None,
                    'bbox_xywh_normalized':obj['bbox_xywh_normalized'], 'bbox_xyxy':bounds,
                    'fine_annotation_known':known, 'fine_annotation_basis':'original_source_label_not_manually_certified',
                    'supervised_fine':False, 'exclusion_reasons':reasons, 'source_flags':sorted(flags),
                    'crop_path':None, 'path':None}
                samples.append(sample); image_record['objects'].append(sample)
                if split=='train' and known:
                    supported_rgbs[fine_id].add(row['rgb_sha256']); supported_groups[fine_id].add(row['group_id'])
            image_records[split].append(image_record)
            labels_before[str(label_path)]=sha(label_path)
            if row.get('label_sha256'): assert labels_before[str(label_path)]==row['label_sha256']
    enough={cid for cid,rgbs in supported_rgbs.items() if len(rgbs)>=args.min_train_images}
    parent_counts=Counter(parents[cid] for cid in enough)
    enabled=sorted(cid for cid in enough if parent_counts[parents[cid]]>=args.min_children_per_parent)
    class_index={cid:i for i,cid in enumerate(enabled)}
    classes=[{'id':class_index[cid], 'name':canonical[cid]['name'], 'parent_id':parents[cid],
              'parent_name':coarse_mapping['classes'][parents[cid]], 'canonical_class_id':cid,
              'train_image_count':len(supported_rgbs[cid]), 'train_images':len(supported_rgbs[cid]),
              'train_group_count':len(supported_groups[cid]), 'enabled':True} for cid in enabled]
    assert len(classes)>1
    for sample in samples:
        cid=sample['canonical_class_id']
        if sample['fine_annotation_known']:
            if cid not in enough: sample['exclusion_reasons'].append('insufficient_distinct_train_images')
            elif cid not in class_index: sample['exclusion_reasons'].append('insufficient_supported_children_for_parent')
            else:
                sample['class_id']=class_index[cid]; sample['supervised_fine']=True
    for split,records in image_records.items():
        for image_record in records:
            materialize=[sample for sample in image_record['objects'] if sample['supervised_fine'] or split!='train']
            if not materialize: continue
            path=Path(image_record['image_path']); sources_before[str(path)]=sha(path)
            if sources_before[str(path)]!=image_record['image_sha256']: raise ValueError(f'Image SHA changed: {path}')
            with Image.open(path) as opened:
                original=opened.convert('RGB')
                rgb=ImageOps.exif_transpose(opened).convert('RGB')
                if rgb.size!=original.size or rgb.tobytes()!=original.tobytes():
                    raise ValueError('EXIF orientation requires explicit box transformation before cropping')
                expected_rgb=hashlib.sha256(f'RGB:{rgb.width}x{rgb.height}:'.encode()+rgb.tobytes()).hexdigest()
                if expected_rgb!=image_record['rgb_sha256']: raise ValueError('RGB hash changed')
                for sample in materialize:
                    target=output/'crops'/split/(sample['id']+'.jpg'); target.parent.mkdir(parents=True,exist_ok=True)
                    original.crop(tuple(sample['bbox_xyxy'])).save(target,quality=95,subsampling=0)
                    sample.update(crop_path=str(target.resolve()),path=str(target.resolve()),crop_sha256=sha(target),
                                  crop_width=sample['bbox_xyxy'][2]-sample['bbox_xyxy'][0],
                                  crop_height=sample['bbox_xyxy'][3]-sample['bbox_xyxy'][1])
    splits={split:[sample for sample in samples if sample['split']==split] for split in split_rows}
    counts=[]
    for cid,cat in sorted(canonical.items()):
        counts.append({'canonical_class_id':cid,'name':cat['name'],'parent_id':parents[cid],
            'enabled':cid in class_index,'class_id':class_index.get(cid),
            'train_image_count':len(supported_rgbs[cid]),'train_group_count':len(supported_groups[cid]),
            'meets_train_support':cid in enough,'source_flags':cat['flags'],
            'splits':{split:{'all_source_objects':sum(s['canonical_class_id']==cid for s in rows),
                            'known_fine_objects':sum(s['canonical_class_id']==cid and s['fine_annotation_known'] for s in rows),
                            'supervised_objects':sum(s['canonical_class_id']==cid and s['supervised_fine'] for s in rows)}
                      for split,rows in splits.items()}})
    for path,value in {**initial_hashes,**labels_before,**sources_before}.items():
        if sha(path)!=value: raise RuntimeError(f'Input changed during preparation: {path}')
    summary={split:{'source_images':len(split_rows[split]),'all_source_objects':len(rows),
                    'known_fine_objects':sum(s['fine_annotation_known'] for s in rows),
                    'supervised_fine_objects':sum(s['supervised_fine'] for s in rows),
                    'supervised_fine_images':len({s['rgb_sha256'] for s in rows if s['supervised_fine']}),
                    'unknown_or_unsupported_objects':sum(not s['supervised_fine'] for s in rows),
                    'materialized_crops':sum(s['crop_path'] is not None for s in rows)} for split,rows in splits.items()}
    audit={'status':'passed','protected_source_ids':sorted(protected_ids),'protected_rgb_sha256':sorted(protected_rgb),
           'protected_group_ids':sorted(protected_groups),'excluded_train_images':excluded_images,
           'split_group_disjoint':True,'split_rgb_disjoint':True,'val_test_not_in_original_coarse_train':True,
           'fixed_original_val59_test59_preserved':True,'legacy14_and_related_groups_excluded':True,
           'input_files_sha256':initial_hashes,'verified_image_sha256':sources_before,'verified_source_label_sha256':labels_before,
           'all_checked_original_files_unchanged':True,'coarse_weights_or_predictions_modified':False}
    manifest={'schema_version':1,'created_at':datetime.now(timezone.utc).isoformat(),
        'status':'prepared_source_label_supervised_fine_crop_pilot_not_completed_human_annotation_v2',
        'scope':'Freeze successful six-class detector; learn additional source-named fine labels only on eligible GT crops',
        'config':{'min_train_images':args.min_train_images,'min_children_per_parent':args.min_children_per_parent,
                  'min_crop_side':args.min_crop_side,'padding':0,'crop_rule':'floor left/top, ceil right/bottom, clamp to image; no resizing'},
        'classes':classes,'class_names':[c['name'] for c in classes],'nc':len(classes),
        'coarse_classes':coarse_mapping['classes'],'summary':summary,'splits':splits,'samples':samples,
        'images':image_records,'source_files_sha256':initial_hashes,
        'support_decision_uses':'training source labels and distinct RGB image counts only; no validation/test scores',
        'annotation_policy':{'pseudo_labels_added':0,'source_labels_modified':0,'human_annotation_v2_completed':False,
            'unknown_fine_targets':'class_id null; never train as negatives or guess fine ground truth',
            'unsupported_known_fine':'retained in validation/test audit and parent coverage; not silently dropped',
            'fixed_coarse_predictions':'Fine outputs are optional attributes; never change box, coarse class or coarse confidence'},
        'limitations':['Named source labels remain unverified; this is a conservative pilot, not complete hierarchy annotation.',
          'Ingredient and preparation names may still have residual semantic overlap requiring manual review.',
          'Classes below minimum support or parents with fewer than two supported children retain coarse-only output.',
          'Threshold calibration uses fixed original validation images only; test labels do not set thresholds.',
          'Only known GT crops supervise the classifier; this does not teach new box localization.']}
    save(output/'manifest.json',manifest); save(output/'class_counts.json',{'classes':counts,'summary':summary})
    save(output/'validation_checks.json',audit)
    save(output/'images.json',image_records)
    for split,rows in splits.items(): save(output/(split+'.json'),rows)
    rows=''.join(f'<tr><td>{html.escape(c["name"])}</td><td>{c["parent_name"]}</td><td>{c["train_image_count"]}</td><td>{c["train_group_count"]}</td></tr>' for c in classes)
    report=f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>階層細分類資料準備</title><style>body{{font:16px/1.7 system-ui;max-width:1100px;margin:32px auto;padding:0 20px;color:#17392b;background:#f3f7f4}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:9px;border:1px solid #cfdbd4}}.note{{background:#fff0d7;padding:15px}}a{{color:#136346}}</style>
<h1>保留六類偵測，新增細分類：來源標籤試跑資料</h1><p class="note">這份資料未完成全量人工階層補標。僅使用可明確對應粗類、沒有已知語意疑義且訓練樣本達門檻的原始細名稱；未知標籤不會當作負例。原始資料與模型未修改。</p>
<p>固定原59張驗證、59張測試與所有關聯群組；原fulltrain1727張移除{len(excluded_images)}張受保護圖片，留下{len(train_rows)}張候選來源圖。其中實際可供細分類訓練：{summary['train']['supervised_fine_images']}張原圖、{summary['train']['supervised_fine_objects']}個裁切框。</p>
<p>{len(enough)}類達到每類至少{args.min_train_images}張不同RGB訓練圖；{len(classes)}類的粗類父節點有至少{args.min_children_per_parent}個可訓練子類，啟用這{len(classes)}類。其餘保留粗類輸出。</p>
<p>驗證保留全部{summary['val']['all_source_objects']}個來源GT框，測試保留全部{summary['test']['all_source_objects']}個來源GT框，包括未支援細類與粗標籤；計分不可只挑已支援細類宣稱全量表現。</p>
<table><tr><th>細名稱</th><th>粗類</th><th>不同訓練圖片</th><th>訓練群組</th></tr>{rows}</table>
<p><a href="manifest.json">完整訓練介面與資料清單</a> · <a href="class_counts.json">全部149類支持數</a> · <a href="validation_checks.json">切分與原始檔案稽核</a></p></html>'''
    (output/'report.html').write_text(report,encoding='utf-8')
    print(json.dumps({'output':str(output),'enabled_classes':len(classes),'train_support_classes':len(enough), 'summary':summary},ensure_ascii=True,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coarse-dataset',type=Path,required=True,
                        help='Audited six-class dataset directory containing manifest/class_mapping/legacy_holdout JSON')
    parser.add_argument('--full-dataset',type=Path,required=True,
                        help='Audited full-source dataset directory containing manifest/class_mapping/validation_checks JSON')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--min-train-images',type=int,default=5)
    parser.add_argument('--min-children-per-parent',type=int,default=2)
    parser.add_argument('--min-crop-side',type=int,default=8)
    args=parser.parse_args()
    if args.min_train_images<1 or args.min_children_per_parent<2 or args.min_crop_side<1:
        parser.error('Minimum support/size must be positive; at least two children per parent')
    build(args)
