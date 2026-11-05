"""Attach a fine crop classifier to immutable saved coarse detections.

Calibrate only on original held-out validation detector crops. The fixed test
predictions are then enriched, never replaced, suppressed, or rescored. A small
optional live before/after canary separately checks the original COCO80 and
successful six-class shared detector. This is not a new full-detector benchmark.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import contextlib
import copy
import html
import io
import json
import math
import os
from pathlib import Path
import time

from bento_research.common import digest, now, save


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def core(row):
    return {key: row[key] for key in ('image_id', 'category_id', 'bbox', 'score')}


def iou(a, b):
    intersection = max(0., min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0])) * max(0., min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))
    return intersection / max(1e-12, a[2]*a[3]+b[2]*b[3]-intersection)


def crop_box(image, box):
    """Match the dataset exact-box crop policy; no context expansion."""
    x, y, w, h = box
    region = (max(0, math.floor(x)), max(0, math.floor(y)),
              min(image.width, math.ceil(x+w)), min(image.height, math.ceil(y+h)))
    if region[2] <= region[0] or region[3] <= region[1]:
        raise ValueError('Degenerate detector crop')
    return image.crop(region)


def normalize_split_samples(manifest, split):
    rows = manifest['splits'][split]
    if isinstance(rows, list) and (not rows or isinstance(rows[0], dict)):
        return rows
    all_samples = manifest.get('samples', [])
    by_id = {r['crop_id']: r for r in all_samples}
    return [by_id[key] for key in rows]


def known_fine_annotations(dataset, split, document, old_manifest):
    """Join known fine targets to original coarse GT by source and geometry.

    Unsupported fine classes are kept as known targets, never hidden from
    confidence calibration. Broad/uncertain source labels remain unknown.
    """
    source_to_image = {old_manifest[im['file_name']]['source_id']: im for im in document['images']}
    by_image = defaultdict(list)
    for annotation in document['annotations']:
        by_image[annotation['image_id']].append(annotation)
    known = {}
    for row in normalize_split_samples(dataset, split):
        if not row.get('fine_annotation_known', row.get('class_id') is not None):
            continue
        source = row['source_id']
        if source not in source_to_image:
            raise ValueError(f'{split} fine sample does not belong to locked coarse split: {source}')
        im = source_to_image[source]
        x, y, w, h = row['bbox_xywh_normalized']
        box = [x*im['width'], y*im['height'], w*im['width'], h*im['height']]
        candidates = [ann for ann in by_image[im['id']] if iou(ann['bbox'], box) >= .999
                      and (row.get('parent_id') is None or ann['category_id'] == row['parent_id'])]
        if len(candidates) != 1:
            raise ValueError(f'Fine target must match one original coarse GT: {source}/{row.get("crop_id")}')
        ann = candidates[0]
        if ann['id'] in known:
            raise ValueError('Duplicate fine annotation for the same original coarse GT')
        known[ann['id']] = {'class_id': row.get('class_id'), 'class_name': row['class_name'],
                            'parent_id': ann['category_id'], 'source_id': source,
                            'supported_by_head': row.get('class_id') is not None}
    return known


def detection_matches(document, predictions):
    """COCO's own score-ordered one-to-one matching at fixed conf .25/IoU .5."""
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(); gt.dataset = copy.deepcopy(document); gt.createIndex()
        dt = gt.loadRes(copy.deepcopy(predictions))
        evaluator = COCOeval(gt, dt, 'bbox')
        evaluator.params.iouThrs = np.array([.5]); evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
    matches = {}
    for row in evaluator.evalImgs:
        if row is None or list(row['aRng']) != list(evaluator.params.areaRng[0]):
            continue
        for detection_id, score, gt_id, ignored in zip(row['dtIds'], row['dtScores'], row['dtMatches'][0], row['dtIgnore'][0]):
            if score >= .25 and gt_id and not ignored:
                matches[int(detection_id)-1] = int(gt_id)
    return matches


def load_split(args, dataset, split):
    folder = args.coarse_evaluation / split
    document = read(folder/'ground_truth.json')
    records = read(folder/'image_predictions.json')
    predictions = [core(row) for row in read(folder/'predictions.json')]
    from_records = [core(row) for record in records for row in record['predictions']]
    if from_records != predictions:
        raise ValueError('Saved image and flat prediction files disagree')
    old_manifest = {Path(row['image']).name: row for row in read(args.coarse_dataset/'manifest.json')['images'] if row['split'] == split}
    if len(records) != 59 or {row['file_name'] for row in records} != set(old_manifest):
        raise ValueError('Expected the original fixed 59-image validation/test split')
    for row in records:
        # Resolve through the supplied dataset so saved predictions may be moved
        # together with their original dataset without retaining machine paths.
        image_path = Path(old_manifest[row['file_name']]['image'])
        row['path'] = str((image_path if image_path.is_absolute() else args.coarse_dataset/image_path).resolve())
        if digest(Path(row['path'])) != row['sha256']:
            raise ValueError('Original evaluation image changed')
    known = known_fine_annotations(dataset, split, document, old_manifest)
    return {'document': document, 'records': records, 'predictions': predictions,
            'known': known, 'matches': detection_matches(document, predictions),
            'source_metrics': read(folder/'metrics.json'),
            'source_sha256': {name: digest(folder/name) for name in ('ground_truth.json','predictions.json','image_predictions.json','metrics.json')}}


def candidate_predictions(refiner, split_data):
    from PIL import Image
    by_image = defaultdict(list)
    for index, prediction in enumerate(split_data['predictions']):
        if prediction['score'] >= .25:
            by_image[prediction['image_id']].append((index, prediction))
    candidates = {}
    for index, image_record in enumerate(split_data['records'], 1):
        rows = by_image[image_record['id']]
        if rows:
            with Image.open(image_record['path']) as image:
                image = image.convert('RGB')
                crops = [crop_box(image, row['bbox']) for _, row in rows]
            output = refiner.predict_crops(crops, [row['category_id'] for _, row in rows])
            if len(output) != len(rows):
                raise ValueError('Refiner returned the wrong crop count')
            for (prediction_index, prediction), result in zip(rows, output):
                if int(result['parent_id']) != prediction['category_id']:
                    raise ValueError('Refiner changed a parent category')
                confidence = result.get('confidence')
                if candidate_name(result) is not None:
                    if confidence is None or not math.isfinite(float(confidence)) or not 0 <= float(confidence) <= 1:
                        raise ValueError('Invalid fine confidence')
                elif confidence is not None:
                    raise ValueError('An absent fine candidate must not have a fabricated confidence')
                candidates[prediction_index] = result
        print(json.dumps({'stage': 'refine_saved_detector_crops', 'image': index, 'total': len(split_data['records'])}), flush=True)
    return candidates


def candidate_name(row):
    return row.get('candidate_class_name', row.get('fine_class_name'))


def candidate_id(row):
    return row.get('candidate_class_id', row.get('fine_class_id'))


def calibrate(classes, split_data, candidates, target_precision, minimum):
    children = Counter(int(row['parent_id']) for row in classes)
    parents = {}
    for parent in range(6):
        labeled = []
        for index, candidate in candidates.items():
            gt = split_data['known'].get(split_data['matches'].get(index))
            if gt is None or candidate['parent_id'] != parent:
                continue
            labeled.append({'prediction_index': index, 'confidence': candidate['confidence'],
                            'has_candidate': candidate_name(candidate) is not None,
                            'correct': candidate_name(candidate) == gt['class_name'],
                            'gt_class': gt['class_name'], 'gt_supported': gt['supported_by_head']})
        row = {'enabled': False, 'threshold': 1., 'reason': 'Insufficient validated detector-crop evidence',
               'trained_children': children[parent], 'known_fine_matched_validation_predictions': len(labeled),
               'unsupported_known_fine_validation_predictions': sum(not item['gt_supported'] for item in labeled),
               'selected_accepted': 0, 'selected_correct': 0, 'observed_precision': None, 'grid': []}
        if children[parent] < 2:
            row['reason'] = 'Fewer than two trained child classes'
        else:
            for threshold in (.5, .6, .7, .8, .85, .9, .95, .98, .99, 1.):
                accepted = [item for item in labeled if item['has_candidate'] and item['confidence'] is not None and item['confidence'] >= threshold]
                correct = sum(item['correct'] for item in accepted)
                precision = correct/len(accepted) if accepted else None
                row['grid'].append({'threshold': threshold, 'accepted': len(accepted), 'correct': correct, 'precision': precision})
            feasible = [value for value in row['grid'] if value['accepted'] >= minimum and value['precision'] >= target_precision]
            if feasible:
                selected = sorted(feasible, key=lambda value: (-value['accepted'], -value['precision'], value['threshold']))[0]
                row.update(enabled=True, threshold=selected['threshold'], reason='Selected using validation detector crops only',
                           selected_accepted=selected['accepted'], selected_correct=selected['correct'], observed_precision=selected['precision'])
        parents[str(parent)] = row
    return {'schema_version': 1, 'scope': 'detector_val_crops', 'created_at': now(), 'parents': parents,
            'target_observed_precision': target_precision, 'minimum_accepted_validation_predictions': minimum,
            'selection_rule': 'Maximum accepted count meeting empirical precision; then precision; then smallest threshold',
            'unsupported_fine_gt_included_as_incorrect': True, 'unknown_broad_gt_excluded_from_fine_accuracy': True,
            'test_used_for_selection': False,
            'limitation': 'Small source-annotated validation sample; empirical target is not a guaranteed deployment precision'}


def enrich(split_data, candidates, thresholds):
    enriched = []
    for index, original in enumerate(split_data['predictions']):
        row = copy.deepcopy(original)
        candidate = candidates.get(index)
        if candidate is None:
            fine = {'accepted': False, 'reason': 'Coarse prediction is below the fixed display threshold',
                    'fine_class_name': None, 'fine_class_id': None}
        else:
            policy = thresholds['parents'][str(original['category_id'])]
            accepted = bool(policy['enabled'] and candidate_name(candidate) is not None and candidate['confidence'] >= policy['threshold'])
            fine = {**candidate, 'accepted': accepted,
                    'fine_class_name': candidate_name(candidate) if accepted else None,
                    'fine_class_id': candidate_id(candidate) if accepted else None,
                    'reason': 'Validation-calibrated fine label added' if accepted else 'Coarse label retained; fine evidence insufficient',
                    'deployment_threshold': policy['threshold'], 'parent_enabled': policy['enabled']}
        row['fine'] = fine
        if core(row) != original:
            raise AssertionError('Fine refinement changed coarse predictions')
        enriched.append(row)
    return enriched


def fine_scores(split_data, enriched):
    counts = Counter()
    for index, prediction in enumerate(enriched):
        if prediction['score'] < .25:
            continue
        counts['displayed_coarse_predictions'] += 1
        fine = prediction['fine']; accepted = fine['accepted']
        if accepted: counts['accepted_fine_predictions'] += 1
        gt_id = split_data['matches'].get(index)
        if gt_id is None:
            counts['unmatched_coarse_predictions'] += 1
            if accepted: counts['accepted_fine_unmatched_coarse_predictions'] += 1
            continue
        gt = split_data['known'].get(gt_id)
        if gt is None:
            counts['matched_but_fine_gt_unknown'] += 1
            if accepted: counts['accepted_fine_with_unknown_gt'] += 1
            continue
        counts['matched_known_fine_gt'] += 1
        if accepted:
            counts['accepted_on_known_fine_gt'] += 1
            counts['correct_accepted_fine'] += fine['fine_class_name'] == gt['class_name']
            counts['wrong_accepted_fine'] += fine['fine_class_name'] != gt['class_name']
        else:
            counts['coarse_fallback_on_known_fine_gt'] += 1
    known_count = len(split_data['known'])
    return {**dict(counts), 'known_fine_gt_total': known_count,
            'known_fine_gt_supported_by_head': sum(row['supported_by_head'] for row in split_data['known'].values()),
            'fine_precision_on_matched_known_gt': counts['correct_accepted_fine']/counts['accepted_on_known_fine_gt'] if counts['accepted_on_known_fine_gt'] else None,
            'end_to_end_known_fine_recall': counts['correct_accepted_fine']/known_count if known_count else None,
            'fine_coverage_on_matched_known_gt': counts['accepted_on_known_fine_gt']/counts['matched_known_fine_gt'] if counts['matched_known_fine_gt'] else None,
            'definition': 'End-to-end recall requires coarse localization, parent match, accepted exact fine name; denominator includes unsupported known fine GT',
            'limitation': 'Fine precision is conditional on matched known fine GT; unmatched/unknown outputs reported separately, not a complete fine-detection AP'}


def live_canary(args, refiner, validation):
    if args.live_canary_count == 0:
        return {'status': 'not_run', 'scope': 'Only saved coarse predictions reused; no fresh model-equivalence claim'}
    import torch
    from PIL import Image
    from verify_bento_frozen_bundle import preprocess, decode, tree_hash
    from bento_research.frozen_yolo_branches import FrozenYOLO26Branches
    from evaluate_bento_shared_bundle import detection_rows
    model = FrozenYOLO26Branches.load_bundle(args.bundle, device=args.device).eval()
    checks = []
    for record in validation['records'][:args.live_canary_count]:
        tensor, original = preprocess(Path(record['path']), args.device)
        with torch.inference_mode(): before = model(tensor, branch='both')
        reference = {branch: tree_hash(before[branch]) for branch in ('old','new')}
        before_rows = detection_rows(decode(before['new'], original), record['id'], record['width'], record['height'], dict(enumerate(range(6))))
        with Image.open(record['path']) as im:
            image = im.convert('RGB')
            shown = [row for row in before_rows if row['score'] >= .25]
            if shown:
                refiner.predict_crops([crop_box(image,row['bbox']) for row in shown], [row['category_id'] for row in shown])
        with torch.inference_mode(): after = model(tensor, branch='both')
        row = {'image': record['file_name'], 'sha256': record['sha256'],
               'coco80_raw_before_after_bitwise_equal': reference['old'] == tree_hash(after['old']),
               'coarse6_raw_before_after_bitwise_equal': reference['new'] == tree_hash(after['new']),
               'fresh_coarse_rows_equal_saved_reference': before_rows == [core(item) for item in record['predictions']],
               'coarse_rows_before_after_equal': before_rows == detection_rows(decode(after['new'], original),record['id'],record['width'],record['height'],dict(enumerate(range(6))))}
        checks.append(row)
        if not all(row[key] for key in ('coco80_raw_before_after_bitwise_equal',
                                       'coarse6_raw_before_after_bitwise_equal',
                                       'fresh_coarse_rows_equal_saved_reference',
                                       'coarse_rows_before_after_equal')):
            save(args.output/'live_canary_failed.json', checks)
            raise RuntimeError('Live frozen-path canary failed; see audit')
    del model
    if args.device.startswith('cuda'): torch.cuda.empty_cache()
    return {'status':'passed','images':len(checks),'checks':checks,
            'scope':'Fresh before/after refinement check on fixed first validation inputs; not a full COCO or all-image live benchmark'}


def gallery(split_data, enriched, output, split):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.load_default(size=14)
    for font_path in ('C:/Windows/Fonts/msjh.ttc', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(font_path).is_file():
            font = ImageFont.truetype(font_path,14)
            break
    names = {row['id']:row['name'] for row in split_data['document']['categories']}
    by_image = defaultdict(list)
    for row in enriched:
        if row['score'] >= .25: by_image[row['image_id']].append(row)
    cards = []
    for record in split_data['records']:
        original = Image.open(record['path']).convert('RGB'); original.thumbnail((760,560))
        scale = original.width/record['width']; panels=[]
        for title, show_fine in [('Frozen successful coarse detector',False),('Same boxes + optional fine labels',True)]:
            panel=Image.new('RGB',(original.width,original.height+46),'#eef4f0');panel.paste(original,(0,46));draw=ImageDraw.Draw(panel)
            draw.text((8,10),title,font=font,fill='#163e2e')
            for row in by_image[record['id']]:
                x,y,w,h=row['bbox'];x*=scale;y=y*scale+46;w*=scale;h*=scale
                accepted=show_fine and row['fine']['accepted'];color='#156ca0' if accepted else '#17764d'
                draw.rectangle((x,y,x+w,y+h),outline=color,width=2)
                label=f"{names[row['category_id']]} {row['score']:.2f}"
                if show_fine:
                    label += ' / '+row['fine']['fine_class_name'] if accepted else ' / coarse only'
                while len(label)>3 and draw.textlength(label,font=font)>panel.width-10:label=label[:-2]
                tx=min(max(2,x),panel.width-draw.textlength(label,font=font)-4);ty=max(46,y-19)
                draw.rectangle(draw.textbbox((tx,ty),label,font=font),fill=color);draw.text((tx,ty),label,font=font,fill='white')
            panels.append(panel)
        canvas=Image.new('RGB',(sum(p.width for p in panels)+12,max(p.height for p in panels)),'white')
        canvas.paste(panels[0],(0,0));canvas.paste(panels[1],(panels[0].width+12,0))
        relative=f'{split}/images/{record["file_name"]}';target=output/relative;target.parent.mkdir(parents=True,exist_ok=True);canvas.save(target,quality=90)
        cards.append({'image':relative,'file_name':record['file_name'],'image_id':record['id'],
                      'coarse_predictions':len(by_image[record['id']]),'fine_added':sum(row['fine']['accepted'] for row in by_image[record['id']])})
    return cards


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--refiner',type=Path,required=True)
    parser.add_argument('--coarse-dataset',type=Path,required=True,
                        help='Original audited six-class dataset directory containing manifest.json')
    parser.add_argument('--coarse-weights',type=Path,required=True)
    parser.add_argument('--coarse-evaluation',type=Path,required=True,
                        help='Saved original shared-detector evaluation directory with val/ and test/')
    parser.add_argument('--bundle',type=Path,
                        help='Original immutable shared bundle, required when live-canary-count is positive')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0',choices=['cpu','cuda:0'])
    parser.add_argument('--live-canary-count',type=int,default=3)
    parser.add_argument('--target-precision',type=float,default=.9)
    parser.add_argument('--minimum-calibration-count',type=int,default=5)
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args(argv)
    if not 0 <= args.live_canary_count <= 59 or not 0 < args.target_precision <= 1 or args.minimum_calibration_count < 1:
        parser.error('Invalid calibration/canary configuration')
    if args.live_canary_count and args.bundle is None:
        parser.error('--bundle is required for live canary checks; use --live-canary-count 0 for saved-output checks only')
    for key in ('dataset','refiner','coarse_dataset','coarse_weights','coarse_evaluation','output'):
        setattr(args,key,getattr(args,key).resolve())
    if args.bundle is not None:
        args.bundle=args.bundle.resolve()
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('Use a new/empty output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    torch.set_num_threads(args.threads);torch.manual_seed(42);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from bento_research.hierarchical_refiner import load_refiner
    from evaluate_bento_shared_bundle import score_native
    started=time.perf_counter();weight_hash=digest(args.coarse_weights)
    dataset=read(args.dataset)
    save(args.output/'protocol.json',{'created_at':now(),'dataset_sha256':digest(args.dataset),'coarse_weights_sha256':weight_hash,
         'classifier_config_sha256':digest(args.refiner/'config.json'),'target_validation_precision':args.target_precision,
         'minimum_calibration_count':args.minimum_calibration_count,'coarse_threshold':.25,'iou_threshold':.5,
         'test_not_used_for_threshold_selection':True,'no_extra_nms_or_score_changes':True})
    refiner=load_refiner(args.refiner,args.coarse_weights,device=args.device)
    validation=load_split(args,dataset,'val')
    val_candidates=candidate_predictions(refiner,validation)
    thresholds=calibrate(dataset['classes'],validation,val_candidates,args.target_precision,args.minimum_calibration_count)
    thresholds['provenance'] = {'dataset_sha256':digest(args.dataset),
         'coarse_checkpoint_sha256':weight_hash, 'classifier_config_sha256':digest(args.refiner/'config.json'),
         'classifier_head_sha256':digest(args.refiner/'head.safetensors'),
         'validation_source_sha256':validation['source_sha256']}
    for key in ('dataset_sha256','coarse_checkpoint_sha256','classifier_config_sha256','classifier_head_sha256'):
        thresholds[key] = thresholds['provenance'][key]
    save(args.output/'deployment_thresholds.json',thresholds)
    if (args.refiner/'thresholds.json').exists():
        save(args.output/'previous_classifier_thresholds.json',read(args.refiner/'thresholds.json'))
    save(args.refiner/'thresholds.json',thresholds)
    refiner.thresholds=thresholds
    live=live_canary(args,refiner,validation);save(args.output/'live_preservation.json',live)
    result={'status':'completed','created_at':now(),'detector_execution':'All59 val/test scores reuse saved immutable coarse predictions; only declared canaries rerun live',
            'fine_execution':'Fresh fine inference on actual predicted-box crops, not GT crops',
            'deployment_thresholds':thresholds,'live_preservation':live,'splits':{},
            'limitations':['Source-derived partial fine annotation; not fully manually adjudicated ground truth',
                          'Coarse fallback preserves existing detections but does not recover pre-existing missed objects',
                          'Fine accepted precision is conditional on known matched source GT; unknown/unmatched outputs separately counted',
                          'Small validation support does not establish commercial precision or robust generalization']}
    for split,split_data,candidates in [('val',validation,val_candidates),('test',None,None)]:
        if split_data is None:
            split_data=load_split(args,dataset,split);candidates=candidate_predictions(refiner,split_data)
        enriched=enrich(split_data,candidates,thresholds)
        before=score_native(copy.deepcopy(split_data['document']),copy.deepcopy(split_data['predictions']))
        after=score_native(copy.deepcopy(split_data['document']),[core(row) for row in enriched])
        for key in ('map50','map50_95','fixed_025_iou_050'):
            if before[key]!=after[key] or before[key]!=split_data['source_metrics'][key]:raise AssertionError('Coarse scoring differs from original baseline')
        save(args.output/split/'enriched_predictions.json',enriched)
        save(args.output/split/'fine_candidates.json',candidates)
        summary={'images':len(split_data['records']),'coarse_predictions_exactly_preserved':True,'coarse_metrics':after,
                 'fine_metrics':fine_scores(split_data,enriched),'source_sha256':split_data['source_sha256'],
                 'gallery':gallery(split_data,enriched,args.output,split)}
        save(args.output/split/'summary.json',summary);result['splits'][split]=summary
    if digest(args.coarse_weights)!=weight_hash:raise AssertionError('Frozen coarse checkpoint changed')
    result.update(seconds=time.perf_counter()-started,coarse_weights_unchanged=True)
    save(args.output/'evaluation.json',result)
    rows='';panels=''
    for split,values in result['splits'].items():
        coarse=values['coarse_metrics']['fixed_025_iou_050']['micro'];fine=values['fine_metrics']
        precision=fine['fine_precision_on_matched_known_gt'];recall=fine['end_to_end_known_fine_recall']
        rows+=f'<tr><td>{split}</td><td>{coarse["precision"]:.2%}</td><td>{coarse["recall"]:.2%}</td><td>{fine.get("accepted_fine_predictions",0)}</td><td>{"未有可計分輸出" if precision is None else format(precision,".2%")}</td><td>{"無已知細標" if recall is None else format(recall,".2%")}</td></tr>'
        panels+=f'<h2>{split}：固定原框＋細類補充</h2>'+''.join(f'<figure><a href="{card["image"]}"><img loading="lazy" src="{card["image"]}"></a><figcaption>{html.escape(card["file_name"])} · 原框 {card["coarse_predictions"]} · 細類補充 {card["fine_added"]}</figcaption></figure>' for card in values['gallery'])
    enabled=[parent for parent,value in thresholds['parents'].items() if value['enabled']]
    page=f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>保留六類偵測＋細分類實驗</title><style>body{{font:16px/1.7 system-ui;margin:24px auto;max-width:1500px;padding:0 20px;background:#f3f7f4;color:#183b2e}}table{{width:100%;border-collapse:collapse;background:white}}td,th{{padding:10px;border:1px solid #c8d7ce}}figure{{background:white;padding:10px;margin:20px 0}}img{{width:100%;height:auto}}.note{{padding:16px;background:#fff0d0}}a{{color:#176b50}}</style>
<h1>保留成功的六類偵測，再增加細分類</h1><p>原本的框、粗類、分數全部保留。細分類沒有把握時顯示 coarse only；沒有額外NMS、改分數或刪框。</p>
<p class="note">59張驗證與59張測試的粗類分數來自已保存的原模型預測，沒有冒充重新跑完整偵測器；細類則實際對這些預測框裁切後執行。另做 {args.live_canary_count} 張現場推論前後檢查，狀態 {live['status']}。細標仍是來源標籤，尚未全量人工校正。</p>
<p>可啟用細類的父類數：{len(enabled)}／6。閾值只由驗證集預測框決定，至少{args.minimum_calibration_count}個有已知細標的接受結果、觀察準確率達{args.target_precision:.0%}；小樣本結果不是商用保證。若全部未通過，保留全部粗類框並據實呈現。</p>
<table><tr><th>集合</th><th>粗類P（保持原值）</th><th>粗類R（保持原值）</th><th>補充細類框</th><th>細類P＊</th><th>已知細類端到端Recall</th></tr>{rows}</table>
<p>＊只在已匹配且來源有已知細標的GT上計算；沒有細標／沒有匹配的輸出另列於JSON，不能當成完整細類偵測AP。缺少訓練支持的已知細類仍包含在Recall分母。</p><p><a href="evaluation.json">完整指標與限制</a> · <a href="deployment_thresholds.json">驗證閾值</a> · <a href="live_preservation.json">實際推論一致性檢查</a></p>{panels}</html>'''
    (args.output/'report.html').write_text(page,encoding='utf-8')
    print(json.dumps({'status':'completed','report':str(args.output/'report.html'),'fine_enabled_parents':enabled}),flush=True)


if __name__=='__main__':main()
