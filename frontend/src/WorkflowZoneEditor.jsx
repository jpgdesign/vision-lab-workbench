import React, { useEffect, useRef, useState } from 'react';
import { Check, CornerUpLeft, MousePointer2, Plus, Trash2 } from 'lucide-react';
import { normalizedPoint, polygonError } from './workflowGraph';

export default function WorkflowZoneEditor({ media, zones = [], onChange, disabled }) {
  const [zoneId, setZoneId] = useState('');
  const [name, setName] = useState('區域 1');
  const [points, setPoints] = useState([]);
  const [drawing, setDrawing] = useState(false);
  const [message, setMessage] = useState('');
  const [ratio, setRatio] = useState(16 / 9);
  const svg = useRef(null), drag = useRef(null);
  const available = Array.isArray(zones) ? zones.filter(zone => zone && typeof zone.name === 'string' && Array.isArray(zone.points) && zone.points.every(point => Array.isArray(point) && point.length === 2 && point.every(Number.isFinite))) : [];
  useEffect(() => { setZoneId(''); setPoints([]); setDrawing(false); setMessage(''); }, [media?.id]);
  function choose(id) {
    const zone = available.find(item => item.name === id);
    setZoneId(id); setName(zone?.name || id || `區域 ${available.length + 1}`);
    setPoints(zone?.points || []); setDrawing(false); setMessage('');
  }
  function start() { setZoneId(''); setName(`區域 ${available.length + 1}`); setPoints([]); setDrawing(true); setMessage(''); }
  function addPoint(event) {
    if (disabled || !drawing || event.target !== svg.current) return;
    const point = normalizedPoint(event.clientX, event.clientY, svg.current.getBoundingClientRect());
    if (point) setPoints(previous => [...previous, point]);
  }
  function movePoint(event) {
    if (drag.current == null || disabled) return;
    const point = normalizedPoint(event.clientX, event.clientY, svg.current.getBoundingClientRect());
    if (point) setPoints(previous => previous.map((value, index) => index === drag.current ? point : value));
  }
  function save() {
    const error = polygonError(points);
    if (error) return setMessage(error);
    const title = name.trim();
    if (!title) return setMessage('請為區域命名。');
    if (available.some(zone => zone.name === title && zone.name !== zoneId)) return setMessage('區域名稱不可重複，請使用不同名稱。');
    const previous = available.find(item => item.name === zoneId);
    const zone = { ...previous, name: title, points };
    onChange([...available.filter(item => item.name !== zoneId), zone]);
    setZoneId(title); setDrawing(false); setMessage('區域已儲存；可拖動頂點調整後再儲存。');
  }
  return <section className="wf-zone-editor">
    <div className="wf-section-title"><strong><MousePointer2 size={14}/>在畫面上定義區域</strong><button type="button" className="button ghost small" onClick={start} disabled={disabled || !media}><Plus size={13}/>新增區域</button></div>
    <p className="wf-hint">點擊畫面加入頂點，拖動圓點調整。區域以影像比例儲存，適用不同解析度。</p>
    {(!Array.isArray(zones) || available.length !== zones.length) && <p className="wf-error">區域資料格式不完整，請在 JSON 中提供 name 與 points。</p>}
    {!media ? <div className="wf-empty-small">先在上方選擇圖片或影片。</div> : <div className={`wf-zone-surface ${drawing ? 'is-drawing' : ''}`} style={{ aspectRatio: ratio }}>
      {media.kind === 'video' ? <video key={media.id} src={media.url} muted playsInline preload="auto" onLoadedMetadata={event => { const video = event.currentTarget; if (video.videoWidth && video.videoHeight) setRatio(video.videoWidth / video.videoHeight); video.currentTime = Math.min(0.05, video.duration || 0); }} /> : <img src={media.url} alt="畫區域的原始素材" onLoad={event => setRatio(event.currentTarget.naturalWidth / event.currentTarget.naturalHeight)}/>}
      <svg ref={svg} viewBox="0 0 1000 1000" preserveAspectRatio="none" role="img" aria-label="區域繪製畫面；也可使用下方 JSON 編輯座標" onPointerDown={addPoint} onPointerMove={movePoint} onPointerUp={() => { drag.current = null; }} onPointerCancel={() => { drag.current = null; }}>
        {available.filter(zone => zone.name !== zoneId).map(zone => <polygon key={zone.name} className="saved-zone" points={(zone.points || []).map(point => `${point[0] * 1000},${point[1] * 1000}`).join(' ')} onPointerDown={event => { event.stopPropagation(); if (!drawing && !disabled) choose(zone.name); }}/>) }
        <polygon className="draft-zone" points={points.map(point => `${point[0] * 1000},${point[1] * 1000}`).join(' ')}/>
        {points.map((point, index) => <circle key={index} cx={point[0] * 1000} cy={point[1] * 1000} r="11" vectorEffect="non-scaling-stroke" onPointerDown={event => { event.stopPropagation(); if (disabled) return; drag.current = index; event.currentTarget.setPointerCapture(event.pointerId); }}/>) }
      </svg>
      <span className="wf-zone-mode">{drawing ? `${points.length} 個頂點 · 點擊繼續繪製` : zoneId ? '拖動頂點調整區域' : '點「新增區域」開始'}</span>
    </div>}
    <label className="field"><span>編輯區域</span><select value={zoneId} disabled={disabled} onChange={event => choose(event.target.value)}><option value="">新的區域</option>{available.map(zone => <option key={zone.name} value={zone.name}>{zone.name}</option>)}</select></label>
    <label className="field"><span>區域名稱</span><input value={name} disabled={disabled} onChange={event => setName(event.target.value)} /></label>
    <div className="wf-inline-actions"><button className="button secondary small" disabled={disabled || !points.length} onClick={() => setPoints(previous => previous.slice(0, -1))}><CornerUpLeft size={13}/>退一點</button><button className="button primary small" disabled={disabled || points.length < 3} onClick={save}><Check size={13}/>儲存區域</button>{zoneId && <button className="icon-button danger-hover" aria-label="刪除選取的區域" disabled={disabled} onClick={() => { onChange(available.filter(zone => zone.name !== zoneId)); choose(''); }}><Trash2 size={15}/></button>}</div>
    {message && <p className="wf-hint" role="status">{message}</p>}
  </section>;
}
