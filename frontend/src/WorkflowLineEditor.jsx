import React, { useEffect, useRef, useState } from 'react';
import { MoveHorizontal, RotateCcw } from 'lucide-react';
import { normalizedPoint } from './workflowGraph';
import './WorkflowLineEditor.css';

const finite = (value, fallback) => Number.isFinite(value) ? value : fallback;

export default function WorkflowLineEditor({ media, params = {}, onChange, disabled, sourceMode = 'media' }) {
  const margin = Math.min(.251, Math.max(.002, finite(params.hysteresis, .025) + .001));
  const clamp = value => Math.round(Math.max(margin, Math.min(1 - margin, value)) * 10000) / 10000;
  const value = clamp(finite(params.line_x, .5));
  const [draft, setDraft] = useState(value), [ratio, setRatio] = useState(16 / 9);
  const svg = useRef(null), dragging = useRef(false), latest = useRef(value);
  const top = Math.max(0, Math.min(1, finite(params.line_y_start, 0)));
  const bottom = Math.max(0, Math.min(1, finite(params.line_y_end, 1)));
  const band = finite(params.hysteresis, .025), viewHeight = 1000 / ratio;
  useEffect(() => {
    if (!dragging.current) { setDraft(value); latest.current = value; }
  }, [value]);
  useEffect(() => { setRatio(16 / 9); }, [media?.id]);
  function update(value, save = false) {
    const next = clamp(value); latest.current = next; setDraft(next);
    if (save) onChange('line_x', next);
  }
  function pointer(event) {
    if (disabled || !svg.current) return;
    const point = normalizedPoint(event.clientX, event.clientY, svg.current.getBoundingClientRect());
    if (point) update(point[0]);
  }
  function keyboard(event) {
    if (disabled) return;
    const step = event.shiftKey ? .05 : .01;
    const next = { ArrowLeft: draft - step, ArrowRight: draft + step, Home: margin, End: 1 - margin }[event.key];
    if (next === undefined) return;
    event.preventDefault(); event.stopPropagation(); update(next, true);
  }
  return <section className="wf-zone-editor wf-line-editor">
    <div className="wf-section-title"><strong><MoveHorizontal size={14}/>設定進出判定線</strong><button type="button" className="button ghost small" disabled={disabled} onClick={() => update(.5, true)}><RotateCcw size={12}/>置中</button></div>
    <p className="wf-hint">點擊或拖曳直線調整位置。左 → 右：出去；右 → 左：進來。方向以送進模型的影像為準。</p>
    <div className="wf-zone-surface wf-line-surface" style={{ aspectRatio: ratio }}>
      {media?.url && (media.kind === 'video'
        ? <video key={media.id} src={media.url} muted playsInline preload="metadata" onLoadedMetadata={event => { const video = event.currentTarget; if (video.videoWidth && video.videoHeight) setRatio(video.videoWidth / video.videoHeight); video.currentTime = Math.min(.05, video.duration || 0); }}/>
        : <img key={media.id} src={media.url} alt="進出判定線的參考影像" onLoad={event => { const image = event.currentTarget; if (image.naturalWidth && image.naturalHeight) setRatio(image.naturalWidth / image.naturalHeight); }}/>)}
      <svg ref={svg} viewBox={`0 0 1000 ${viewHeight}`} preserveAspectRatio="none" role="slider" tabIndex={disabled ? -1 : 0} aria-label="進出判定線水平位置" aria-orientation="horizontal" aria-valuemin={Math.round(margin * 100)} aria-valuemax={Math.round((1 - margin) * 100)} aria-valuenow={Math.round(draft * 100)} aria-valuetext={`距離左邊 ${(draft * 100).toFixed(1)}%`} aria-disabled={disabled} onKeyDown={keyboard}
        onPointerDown={event => { if (disabled) return; event.preventDefault(); dragging.current = true; event.currentTarget.setPointerCapture(event.pointerId); pointer(event); }}
        onPointerMove={event => { if (dragging.current) pointer(event); }}
        onPointerUp={event => { if (!dragging.current) return; pointer(event); dragging.current = false; onChange('line_x', latest.current); if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId); }}
        onPointerCancel={() => { dragging.current = false; setDraft(value); latest.current = value; }}>
        <rect className="wf-line-band" x={(draft - band) * 1000} y={top * viewHeight} width={band * 2000} height={Math.max(0, bottom - top) * viewHeight}/>
        <line className="wf-line-guide" x1={draft * 1000} y1={top * viewHeight} x2={draft * 1000} y2={bottom * viewHeight} vectorEffect="non-scaling-stroke"/>
        <circle cx={draft * 1000} cy={(top + bottom) / 2 * viewHeight} r="20" vectorEffect="non-scaling-stroke"/>
        <text className="wf-line-text" x="32" y="55">出去 →</text><text className="wf-line-text" x="968" y="55" textAnchor="end">← 進來</text>
      </svg>
      <span className="wf-zone-mode">{media ? sourceMode === 'media' ? '依素材比例設定' : '參考素材 · 即時影像沿用相同比例' : '未選參考圖 · 可先設定相對位置'} · {(draft * 100).toFixed(1)}%</span>
    </div>
    <label className="field wf-line-range"><span>水平位置 · 距左側 {(draft * 100).toFixed(1)}%</span><input type="range" aria-label="中線水平位置" min={margin} max={1 - margin} step="0.001" value={draft} disabled={disabled} onChange={event => update(Number(event.target.value), true)}/></label>
    <p className="wf-hint">陰影帶內的抖動不計次；跨線後需在另一側穩定出現。可用方向鍵微調，Shift 加方向鍵每次移動 5%。</p>
    <p className="wf-hint">「認真」是你設定的坐姿名稱；需要可見的肩、髖、膝、腳踝。桌子遮擋雙腿時顯示姿態不明。</p>
  </section>;
}
