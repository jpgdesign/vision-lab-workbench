"""Show saved fine-label candidates without changing model acceptance or thresholds."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import html
import json
from pathlib import Path
import textwrap
from urllib.parse import quote

from PIL import Image, ImageDraw, ImageFont


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def font(size):
    for path in ('C:/Windows/Fonts/msjh.ttc', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(path).is_file(): return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def draw_pair(image_path, rows, names, destination):
    title_font, label_font = font(19), font(16)
    with Image.open(image_path) as opened:
        source=opened.convert('RGB'); source.thumbnail((760,560))
        scale=source.width/opened.width
    panel_width=max(520,source.width)
    offset=(panel_width-source.width)//2
    panels=[]
    for raw in (False,True):
        panel=Image.new('RGB',(panel_width,source.height+66),'#f4f6f2');panel.paste(source,(offset,66))
        draw=ImageDraw.Draw(panel)
        draw.text((10,8),'Raw fine candidates - unverified' if raw else 'Original coarse predictions',font=title_font,fill='#193b2e')
        draw.text((10,36),'Candidate score is NOT accuracy' if raw else 'Same boxes, classes and scores retained',font=label_font,fill='#6b5733' if raw else '#3d6653')
        for number,row in enumerate(rows,1):
            x,y,w,h=row['bbox'];x=x*scale+offset;y=y*scale+66;w*=scale;h*=scale
            fine=row['fine']; candidate=fine.get('candidate_class_name')
            color='#b16a00' if raw and candidate else '#557269' if raw else '#17764d'
            draw.rectangle((x,y,x+w,y+h),outline=color,width=3)
            if raw:
                label=f'#{number} {candidate} {fine["confidence"]:.3f}' if candidate else f'#{number} {names[row["category_id"]]} / no fine head'
            else: label=f'#{number} {names[row["category_id"]]} {row["score"]:.3f}'
            maximum=max(18,int((panel_width-12)/8.7))
            lines=textwrap.wrap(label,width=min(43,maximum),break_long_words=True,break_on_hyphens=False)
            label='\n'.join(lines)
            bounds=draw.multiline_textbbox((0,0),label,font=label_font,spacing=1)
            tw,th=bounds[2]-bounds[0],bounds[3]-bounds[1]
            tx=min(max(3,x),max(3,panel_width-tw-7));ty=max(67,y-th-5)
            draw.rectangle((tx-2,ty-2,tx+tw+4,ty+th+5),fill=color)
            draw.multiline_text((tx,ty),label,font=label_font,spacing=1,fill='white')
        panels.append(panel)
    combined=Image.new('RGB',(panel_width*2+12,source.height+66),'white')
    combined.paste(panels[0],(0,0));combined.paste(panels[1],(panel_width+12,0))
    destination.parent.mkdir(parents=True,exist_ok=True);combined.save(destination,quality=91)


def add_navigation(path, target, label):
    if not path.is_file(): return
    page=path.read_text(encoding='utf-8')
    marker='<!-- raw-fine-candidate-navigation -->'
    if marker in page: return
    end=page.find('</h1>')
    if end<0: raise ValueError('Expected report heading')
    nav=f'{marker}<p style="padding:14px;background:#fff0cf;border:2px solid #b57818;border-radius:8px"><a style="color:#824d00;font-weight:700" href="{html.escape(target,quote=True)}">{html.escape(label)}</a>：直接查看模型猜測的細類名稱與分數。</p>'
    page=page[:end+5]+nav+page[end+5:];path.write_text(page,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--coarse-evaluation',type=Path,required=True)
    parser.add_argument('--coarse-dataset',type=Path,help='Optional relocated coarse dataset root')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--add-navigation',action='store_true')
    args=parser.parse_args()
    evaluation=args.evaluation.resolve(); output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    protected=[evaluation/'deployment_thresholds.json']
    if (evaluation.parent/'classifier/thresholds.json').exists():protected.append(evaluation.parent/'classifier/thresholds.json')
    for split in ('val','test'):protected.append(evaluation/split/'enriched_predictions.json')
    before={str(path):digest(path) for path in protected}
    summary={};cards=[]
    for split in ('test','val'):
        doc=read(args.coarse_evaluation/split/'ground_truth.json')
        names={c['id']:c['name'] for c in doc['categories']}
        images=read(args.coarse_evaluation/split/'image_predictions.json')
        enriched=read(evaluation/split/'enriched_predictions.json')
        original=[{k:row[k] for k in ('image_id','category_id','bbox','score')} for row in read(args.coarse_evaluation/split/'predictions.json')]
        if [{k:row[k] for k in ('image_id','category_id','bbox','score')} for row in enriched]!=original:
            raise ValueError('Coarse predictions changed')
        grouped=defaultdict(list)
        for row in enriched:
            if row['score']>=.25:grouped[row['image_id']].append(row)
        all_displayed=[row for rows in grouped.values() for row in rows]
        summary[split]={'images':len(images),'coarse_boxes':len(all_displayed),
                        'raw_fine_candidates':sum(row['fine'].get('candidate_class_name') is not None for row in all_displayed),
                        'accepted_fine':sum(bool(row['fine']['accepted']) for row in all_displayed)}
        for image in images:
            path=Path(image['path'])
            if args.coarse_dataset: path=args.coarse_dataset/split/'images'/image['file_name']
            if digest(path)!=image['sha256']:raise ValueError('Evaluation image changed')
            rows=grouped[image['id']]
            target=output/'images'/split/image['file_name'];draw_pair(path,rows,names,target)
            items=[]
            for number,row in enumerate(rows,1):
                fine=row['fine'];candidate=fine.get('candidate_class_name')
                state='通過顯示門檻' if fine['accepted'] else '未通過顯示門檻' if candidate else '此粗類未啟用細分類頭'
                top=sorted(fine.get('probabilities',{}).items(),key=lambda x:x[1],reverse=True)[:3]
                items.append({'number':number,'coarse':names[row['category_id']],'coarse_score':row['score'],
                              'fine_candidate':candidate,'candidate_score':fine.get('confidence'),
                              'accepted':bool(fine['accepted']),'display_status':state,'top3':top})
            cards.append({'split':split,'file_name':image['file_name'],'image':target.relative_to(output).as_posix(),
                          'image_id':image['id'],'rows':items,'raw_count':sum(x['fine_candidate'] is not None for x in items)})
    def e(value):return html.escape(str(value))
    rendered=[]
    for card in cards:
        rows=''
        for item in card['rows']:
            top='；'.join(f'{e(name)} {score:.3f}' for name,score in item['top3'])
            candidate=e(item['fine_candidate']) if item['fine_candidate'] else '—'
            score=f'{item["candidate_score"]:.3f}' if item['candidate_score'] is not None else '—'
            rows+=f'<tr><td>#{item["number"]}</td><td>{e(item["coarse"])}<small>粗類分數 {item["coarse_score"]:.3f}</small></td><td><strong>{candidate}</strong></td><td>{score}</td><td>{e(item["display_status"])}</td><td>{top or "—"}</td></tr>'
        rendered.append(f'<article data-split="{card["split"]}" data-search="{e(card["file_name"])}" data-fine="{card["raw_count"]}"><h2>{"測試集" if card["split"]=="test" else "驗證集"} · {e(card["file_name"])}</h2><p>{len(card["rows"])} 個粗類框，其中 {card["raw_count"]} 個有細類候選。</p><a href="{card["image"]}"><img loading="lazy" src="{card["image"]}" alt="原粗類框與細分類候選並排圖"></a><details><summary>查看每個框的完整細類名稱、模型分數與前3候選</summary><div class="scroll"><table><tr><th>框</th><th>粗類</th><th>細類候選</th><th>候選分數</th><th>目前狀態</th><th>前3候選</th></tr>{rows or "<tr><td colspan=6>此圖沒有粗類分數達0.25的框。</td></tr>"}</table></div></details></article>')
    data={'scope':'Raw diagnostic fine candidates; not accepted outputs or calibrated accuracy',
          'summary':summary,'images':cards,'preserved_input_sha256':before,
          'postprocessing':'Saved coarse threshold0.25; original boxes/scores unchanged; no additional inference or calibration'}
    (output/'gallery.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    page='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>細分類原始預測｜便當模型</title>
<style>body{font:16px/1.65 system-ui;margin:0;background:#f5f6f2;color:#213c31}main{max-width:1540px;margin:auto;padding:24px}h1{font-size:30px}h2{font-size:20px}.notice{padding:16px;border-left:5px solid #b8740b;background:#fff0d3}.toolbar{position:sticky;top:0;display:flex;gap:14px;flex-wrap:wrap;align-items:center;background:#f5f6f2f5;padding:14px 0;z-index:2}select,input{font:inherit;padding:8px}article{background:white;padding:16px;margin:22px 0;border-radius:10px;border:1px solid #d6ded8}img{width:100%;height:auto;display:block}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #dde4df;text-align:left;padding:9px;vertical-align:top}small{display:block;color:#587064}summary{cursor:pointer;padding:12px 0;font-weight:bold}.scroll{overflow-x:auto}a{color:#835103}[hidden]{display:none!important}@media(max-width:600px){main{padding:12px}h1{font-size:24px}}</style>
<main><h1>細分類原始預測</h1><p class="notice"><strong>這裡會顯示模型猜測的細類，即使尚未通過驗證門檻。</strong>左圖是原本的粗類，右圖橘色為細類候選；灰色代表該粗類沒有啟用細分類頭。候選分數是同一粗類內的模型分數，不能當成辨識正確率。目前最終顯示仍保留粗類。</p>
<p>__COUNTS__ 此頁包含全部圖片，不只挑正確圖片；原框與顯示門檻未改。</p>
<p><a href="../report.html">回到通過門檻後的顯示</a> · <a href="../../report.html">實驗總報告</a> · <a href="gallery.json">完整候選與來源索引</a></p>
<div class="toolbar"><label>資料 <select id="split"><option value="test">測試集59張</option><option value="val">驗證集59張</option><option value="all">全部118張</option></select></label><label><input type="checkbox" id="fine">只看有細類候選的圖片</label><label>檔名 <input id="search" placeholder="搜尋圖片編號"></label><span id="count"></span></div>
'''+''.join(rendered)+'''</main><script>const cards=[...document.querySelectorAll('article')];function filter(){const split=document.getElementById('split').value;const fine=document.getElementById('fine').checked;const query=document.getElementById('search').value.trim().toLowerCase();let count=0;for(const card of cards){const show=(split==='all'||card.dataset.split===split)&&(!fine||Number(card.dataset.fine)>0)&&card.dataset.search.toLowerCase().includes(query);card.hidden=!show;if(show)count++;}document.getElementById('count').textContent=`顯示 ${count} 張`;}for(const id of ['split','fine','search'])document.getElementById(id).addEventListener('input',filter);filter();</script></html>'''
    counts=' '.join(f'{label}{summary[split]["images"]}張：{summary[split]["coarse_boxes"]}個粗類框，{summary[split]["raw_fine_candidates"]}個有細類候選。' for split,label in [('test','測試'),('val','驗證')])
    page=page.replace('__COUNTS__',counts)
    (output/'report.html').write_text(page,encoding='utf-8')
    if args.add_navigation:
        import os
        add_navigation(evaluation/'report.html',quote(Path(os.path.relpath(output/'report.html',evaluation)).as_posix()),'查看細分類原始預測（含未通過門檻）')
        add_navigation(evaluation.parent/'report.html',quote(Path(os.path.relpath(output/'report.html',evaluation.parent)).as_posix()),'查看細分類原始預測（含未通過門檻）')
    if any(digest(Path(path))!=value for path,value in before.items()):raise RuntimeError('Protected evaluation inputs changed')
    print(json.dumps({'report':str(output/'report.html'),'summary':summary,'unchanged_inputs':True},ensure_ascii=True))


if __name__=='__main__':main()
