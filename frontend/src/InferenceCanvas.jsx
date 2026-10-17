import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { Background, ReactFlow, ReactFlowProvider, NodeResizer, useNodesState, useReactFlow } from '@xyflow/react';
import { ArrowDownToLine, ArrowUpFromLine, ChevronDown, Clapperboard, Copy, Expand, Film, Image, Layers2, LoaderCircle, Maximize, Play, Plus, ScanLine, Search, SlidersHorizontal, Square, Trash2, Upload, X, ZoomIn, ZoomOut } from 'lucide-react';
import { activeJob, modelDefaults, optionParts, statusNames, taskNames, uid } from './api';
import { canRunWithWeights, checkpointChoices, checkpointRequest, checkpointUnavailable, selectedCheckpoint as checkpointSelection } from './checkpoints';
import { deviceAvailable } from './systemDevices';

const CanvasContext = createContext(null);
const STORAGE_KEY = 'vision-lab-canvas-v1';
const READABLE_ZOOM = 0.72;
function readStored() { try { const value = JSON.parse(localStorage.getItem(STORAGE_KEY)); return value?.version === 1 ? value : null; } catch { return null; } }
function safeNodes(nodes, models) {
  if (!Array.isArray(nodes) || nodes.length > 24) throw new Error('畫布最多允許 24 個比較區塊。');
  return nodes.map((node, index) => {
    const model = models.find(item => item.id === node?.data?.modelId);
    const params = {};
    for (const param of model?.capabilities?.inference_params || []) { const value = node.data?.params?.[param.key]; if (['string', 'number', 'boolean'].includes(typeof value)) params[param.key] = value; }
    return { id: uid(), type: 'inference', dragHandle: '.node-drag', position: { x: Math.max(-10000, Math.min(10000, Number(node.position?.x) || index * 460)), y: Math.max(-10000, Math.min(10000, Number(node.position?.y) || 30)) }, style: { width: Math.max(370, Math.min(900, Number(node.style?.width) || 420)), height: Math.max(540, Math.min(1200, Number(node.style?.height) || 600)) }, data: { modelId: model?.id || models.find(model => model.inference_supported)?.id || '', params: { ...modelDefaults(model), ...params }, mediaId: typeof node.data?.mediaId === 'string' ? node.data.mediaId : '', jobId: typeof node.data?.jobId === 'string' ? node.data.jobId : null, checkpointJobId: typeof node.data?.checkpointJobId === 'string' ? node.data.checkpointJobId : '', checkpointId: typeof node.data?.checkpointId === 'string' ? node.data.checkpointId : '' } };
  });
}
export function ParameterField({ param, value, onChange, compact = false, disabled = false }) {
  const label = param.label || param.key;
  const unavailableDevice = param.key === 'device' && value && !(param.options || []).map(optionParts).some(option => option.value === value);
  if (param.type === 'boolean') return <label className={`check-field ${compact ? 'compact' : ''}`}><input type="checkbox" disabled={disabled} checked={Boolean(value)} onChange={event => onChange(event.target.checked)} /><span>{label}</span></label>;
  return <label className={`field ${compact ? 'compact' : ''}`}><span>{label}</span>{param.type === 'select' ? <select disabled={disabled} value={value ?? param.default ?? ''} onChange={event => { const selected = (param.options || []).map(optionParts).find(option => String(option.value) === event.target.value); onChange(selected ? selected.value : event.target.value); }}>{unavailableDevice && <option value={value} disabled>{value} · 目前不可用</option>}{(param.options || []).map(option => { const parsed = optionParts(option); return <option key={parsed.value} value={parsed.value}>{parsed.label}</option>; })}</select> : <input disabled={disabled} type={param.type === 'number' ? 'number' : 'text'} value={value ?? param.default ?? ''} min={param.min} max={param.max} step={param.step || 'any'} onChange={event => onChange(param.type === 'number' ? (event.target.value === '' ? '' : Number(event.target.value)) : event.target.value)} />}</label>;
}
function InferenceNode({ id, data, selected }) {
  const context = useContext(CanvasContext);
  const { models, media, jobs, system, confirmLicense, checkpoints: registeredCheckpoints, sharedInput, selectedMedia, updateNode, duplicateNode, removeNode, runNode, cancelJob, nodeOrder } = context;
  const [advanced, setAdvanced] = useState(false);
  const [original, setOriginal] = useState(false);
  const model = models.find(item => item.id === data.modelId);
  const job = jobs.find(item => item.id === data.jobId);
  const active = activeJob(job);
  const result = job?.status === 'completed' ? job.result : null;
  const nextAsset = media.find(item => item.id === (sharedInput ? selectedMedia : data.mediaId));
  const runMediaId = data.runMediaId || job?.media_id;
  const asset = (active || result) && runMediaId ? media.find(item => item.id === runMediaId) || nextAsset : nextAsset;
  const mediaCompatible = !nextAsset || model?.capabilities?.input_types?.includes(nextAsset.kind);
  const checkpoints = checkpointChoices(data.modelId, registeredCheckpoints, jobs);
  const support = canRunWithWeights(model, data, checkpoints);
  const selectedCheckpoint = checkpoints.find(item => item.id === checkpointSelection(data));
  const checkpointMissing = checkpointUnavailable(data, checkpoints);
  const reason = !model ? '請先選擇模型' : !support ? model.inference_reason || model.reason || '此模型目前尚未提供可用的推論介面' : !deviceAvailable(data.params?.device, system) ? '選取的運算裝置目前不可用，請重新選擇。' : checkpointMissing ? '選取的權重已不可用，請重新選擇或在模型與任務檢查路徑。' : !nextAsset ? '請選擇一張圖片或影片' : !mediaCompatible ? `此模型不支援${nextAsset.kind === 'image' ? '單張圖片' : '影片'}輸入` : model.license_url && model.license_accepted !== true && !data.params?.accept_platform_license ? '請先閱讀授權並勾選確認' : '';
  const params = (model?.capabilities?.inference_params || []).filter(param => param.key !== 'accept_platform_license' || model?.license_accepted !== true);
  const visibleParams = advanced ? params : params.filter((param, index) => index < 2 || param.key === 'accept_platform_license');
  const mediaUrl = !original && result?.media_url ? result.media_url : asset?.url;
  const mediaKind = !original && result?.media_url ? result.media_kind : asset?.kind;
  const predictions = result?.predictions || [];
  const trainedClasses = selectedCheckpoint?.training?.classes?.filter(name => typeof name === 'string') || [];
  const suggestedPrompt = selectedCheckpoint?.training?.caption || (trainedClasses.length ? `${trainedClasses.join(' . ')} .` : '');
  return <div className={`inference-node ${active ? 'is-running' : ''}`}>
    <NodeResizer isVisible={selected} minWidth={370} minHeight={550} maxWidth={900} maxHeight={1300} lineClassName="resize-line" handleClassName="resize-handle" />
    <div className="node-drag"><div className="node-heading"><span className="node-number">{String(nodeOrder.indexOf(id) + 1).padStart(2, '0')}</span><div><strong>{model?.name || '選擇模型'}</strong><small>{model ? taskNames[model.task] : '新增推論實驗'}</small></div></div><div className="node-actions nodrag"><button className="icon-button" title="複製區塊" aria-label="複製區塊" onClick={() => duplicateNode(id)}><Copy size={15}/></button><button className="icon-button danger-hover" title="移除區塊" aria-label="移除區塊" onClick={() => removeNode(id)}><X size={16}/></button></div></div>
    <div className="node-config nodrag nowheel">
      <label className="field"><span>模型 <span className="tiny-label">MODEL</span></span><select value={data.modelId} disabled={active} onChange={event => { const next = models.find(item => item.id === event.target.value); updateNode(id, { modelId: event.target.value, params: modelDefaults(next), jobId: null, checkpointJobId: '', checkpointId: '' }); }}>{models.map(item => <option value={item.id} key={item.id}>{item.name}{!item.available ? ' · 未就緒' : !item.inference_supported ? ' · 尚未支援' : ''}</option>)}</select></label>
      {(checkpoints.length > 0 || checkpointSelection(data)) && <label className="field checkpoint-field"><span>模型權重 <span className="tiny-label">CHECKPOINT</span></span><select value={checkpointSelection(data)} disabled={active} onChange={event => updateNode(id, { checkpointId: event.target.value, checkpointJobId: '', jobId: null })}><option value="" disabled={!model?.available}>{model?.available ? '原始預訓練權重' : '原始權重未齊，請選擇微調權重'}</option>{checkpointSelection(data) && !selectedCheckpoint && <option value={checkpointSelection(data)} disabled>先前權重已不可用</option>}{checkpoints.map(item => <option key={item.id} value={item.id} disabled={!item.valid}>{item.source === 'external' ? '外部 · ' : ''}{item.name}{!item.valid ? ' · 路徑失效' : ''}</option>)}</select></label>}
      {trainedClasses.length > 0 && <div className="checkpoint-class-info"><strong>微調類別 · {trainedClasses.length}</strong><p title={trainedClasses.join(' · ')}>{trainedClasses.slice(0, 10).join(' · ')}{trainedClasses.length > 10 ? ` · +${trainedClasses.length - 10}` : ''}</p>{data.modelId === 'grounding_dino_base' && suggestedPrompt && <button className="button ghost small" disabled={active || suggestedPrompt.length > 2000} title={suggestedPrompt.length > 2000 ? '類別文字過長，請在偵測目標欄位選填需要測試的類別。' : suggestedPrompt} onClick={() => updateNode(id, { params: { ...data.params, prompt: suggestedPrompt }, jobId: null })}>套用訓練類別為偵測目標</button>}</div>}
      {!sharedInput && <label className="field"><span>輸入素材</span><select value={data.mediaId || ''} disabled={active} onChange={event => updateNode(id, { mediaId: event.target.value, jobId: null })}><option value="">選擇輸入素材</option>{media.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>}
      <div className="node-params">{visibleParams.map(param => <ParameterField compact disabled={active} key={param.key} param={param} value={data.params?.[param.key]} onChange={value => { updateNode(id, { params: { ...data.params, [param.key]: value }, jobId: null }); if (param.key === 'accept_platform_license' && value === true) confirmLicense(model.id); }}/>)}</div>
      {model?.license_url && <a className="license-link" href={model.license_url} target="_blank" rel="noreferrer">{model.license_accepted === true ? '授權已確認 · 閱讀條款 ↗' : '閱讀模型授權條款 ↗'}</a>}
      {params.length > 2 && <button className="parameter-toggle" onClick={() => setAdvanced(value => !value)}><SlidersHorizontal size={12}/>{advanced ? '收起參數' : `更多參數 (${params.length - 2})`}<ChevronDown size={13} className={advanced ? 'rotated' : ''}/></button>}
    </div>
    <div className="preview-area nodrag nowheel">
      {mediaUrl ? <>{mediaKind === 'video' ? <video src={mediaUrl} controls preload="metadata" /> : <img src={mediaUrl} alt={result && !original ? `${model?.name} 推論結果` : asset?.name || '輸入圖片'} />}<span className={`preview-label ${result && !original ? 'result' : ''}`}>{result && !original ? '推論結果' : '原始輸入 · 尚未套用結果'}</span></> : <div className="empty-preview"><ScanLine size={34} strokeWidth={1.2}/><strong>讓這裡成為你的第一個實驗</strong><span>選擇素材與模型，然後開始推論</span></div>}
      {active && <div className="running-overlay"><LoaderCircle size={25} className="spin"/><span>{statusNames[job.status]}</span><small>{job.message || '等待模型回傳結果'}</small></div>}
      {result?.media_url && <button className="preview-switch" onClick={() => setOriginal(value => !value)}>{original ? '顯示結果' : '查看原圖'}</button>}
      {result && asset && <span className="preview-source" title={`這次結果的輸入：${asset.name}`}>{asset.name}</span>}
    </div>
    <div className="node-result nodrag nowheel">
      {result ? <><div className="result-facts"><span><i className="status-dot"/>推論完成</span>{result.inference_ms != null && <b>{Number(result.inference_ms).toFixed(0)} <small>ms</small></b>}{predictions.length > 0 && <span>{predictions.length} 個預測</span>}{result.frames != null && <span>{result.frames} 幀</span>}</div>{predictions.length > 0 ? <div className="prediction-list">{predictions.slice(0, 12).map((prediction, index) => <span key={index}>{prediction.label}<b>{Number.isFinite(prediction.score) ? `${(prediction.score * 100).toFixed(1)}%` : ''}</b></span>)}{predictions.length > 12 && <span>+{predictions.length - 12}</span>}</div> : <p className="result-summary">{typeof result.summary === 'string' ? result.summary : '執行完成，沒有回傳分類或偵測標籤。'}</p>}</> : job?.status === 'failed' ? <div className="inline-error">{job.error || job.message || '執行失敗，請檢查模型設定與任務紀錄。'}</div> : <p className="result-placeholder">{active ? job.message || '實際執行後顯示辨識結果與耗時' : reason || '準備就緒，開始後顯示真實辨識結果與耗時'}</p>}
    </div>
    <div className="node-footer nodrag"><span className={`job-status ${job?.status || 'idle'}`}><i/>{job ? statusNames[job.status] : '尚未執行'}</span>{active ? <button className="button secondary small" onClick={() => cancelJob(job.id)}><Square size={12}/>取消</button> : <button className="button primary small" disabled={Boolean(reason)} title={reason || '執行這個推論實驗'} onClick={() => runNode(id)}><Play size={13} fill="currentColor"/>開始推論</button>}</div>
  </div>;
}
const nodeTypes = { inference: InferenceNode };

function CanvasInner(props) {
  const { models, media, jobs, system, confirmLicense, checkpoints = [], notify, submitJob, cancelJob, uploadMedia, visible } = props;
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [selectedMedia, setSelectedMedia] = useState('');
  const [sharedInput, setSharedInput] = useState(true);
  const [filter, setFilter] = useState('all');
  const [query, setQuery] = useState('');
  const [uploading, setUploading] = useState(false);
  const [startingAll, setStartingAll] = useState(false);
  const [initialized, setInitialized] = useState(false);
  const fileInput = useRef(null), importInput = useRef(null);
  const flow = useReactFlow();
  const nodesRef = useRef(nodes); nodesRef.current = nodes;
  function fitReadable(items = nodesRef.current) {
    flow.fitView({ nodes: items.slice(0, 2).map(node => ({ id: node.id })), padding: 0.08, minZoom: READABLE_ZOOM, maxZoom: 1, duration: 200 });
  }
  useEffect(() => {
    if (initialized || !models.length) return;
    const stored = readStored();
    const first = media[0]?.id || '';
    if (stored) {
      try { setNodes(safeNodes(stored.nodes, models)); setSelectedMedia(media.find(item => item.id === stored.selectedMedia)?.id || first); setSharedInput(stored.sharedInput !== false); }
      catch { setNodes([]); }
    } else {
      const initialModels = models.filter(model => model.available && model.inference_supported).slice(0, 2);
      setNodes(initialModels.map((model, index) => ({ id: uid(), type: 'inference', dragHandle: '.node-drag', position: { x: 28 + index * 452, y: 30 }, style: { width: 420, height: 590 }, data: { modelId: model.id, mediaId: first, params: modelDefaults(model), jobId: null } })));
      setSelectedMedia(first);
    }
    setInitialized(true);
  }, [models, media, initialized, setNodes]);
  useEffect(() => { if (!selectedMedia && media.length) setSelectedMedia(media[0].id); }, [media, selectedMedia]);
  useEffect(() => { if (initialized) { try { localStorage.setItem(STORAGE_KEY, JSON.stringify({ version: 1, nodes, selectedMedia, sharedInput })); } catch { /* Browser storage may be unavailable. Export remains available. */ } } }, [nodes, selectedMedia, sharedInput, initialized]);
  useEffect(() => { if (visible && initialized) { const timer = setTimeout(() => fitReadable(), 100); return () => clearTimeout(timer); } }, [visible, initialized]);
  const updateNode = useCallback((id, patch) => setNodes(previous => previous.map(node => node.id === id ? { ...node, data: { ...node.data, ...patch } } : node)), [setNodes]);
  function addNode() {
    if (nodes.length >= 24) return notify('畫布最多允許 24 個比較區塊。', true);
    const model = models.find(item => item.available && item.inference_supported) || models[0];
    const rightEdge = nodes.reduce((end, node) => Math.max(end, node.position.x + Number(node.style?.width || 420)), -4);
    const node = { id: uid(), type: 'inference', dragHandle: '.node-drag', position: { x: rightEdge + 32, y: 30 }, style: { width: 420, height: 590 }, data: { modelId: model?.id || '', mediaId: selectedMedia, params: modelDefaults(model), jobId: null }, selected: true };
    setNodes(previous => [...previous.map(item => ({ ...item, selected: false })), node]);
    setTimeout(() => flow.fitView({ nodes: [{ id: node.id }], padding: 0.12, minZoom: READABLE_ZOOM, maxZoom: 1, duration: 250 }), 80);
  }
  function duplicateNode(id) {
    if (nodes.length >= 24) return notify('畫布最多允許 24 個比較區塊。', true);
    const node = nodes.find(item => item.id === id); if (!node) return;
    const copy = { ...node, id: uid(), selected: true, position: { x: node.position.x + Number(node.style?.width || 420) + 28, y: node.position.y + 24 }, data: { ...node.data, params: { ...node.data.params }, jobId: null } };
    setNodes(previous => [...previous.map(item => ({ ...item, selected: false })), copy]); setTimeout(() => flow.fitView({ nodes: [{ id: copy.id }], padding: 0.12, minZoom: READABLE_ZOOM, maxZoom: 1, duration: 200 }), 80);
  }
  function removeNode(id) {
    const node = nodes.find(item => item.id === id), job = jobs.find(item => item.id === node?.data.jobId);
    if (activeJob(job)) { notify('任務仍在執行，請先取消再移除區塊。', true); return; }
    setNodes(previous => previous.filter(item => item.id !== id));
  }
  async function runNode(id) {
    const node = nodesRef.current.find(item => item.id === id); if (!node || activeJob(jobs.find(item => item.id === node.data.jobId))) return;
    const model = models.find(item => item.id === node.data.modelId), mediaId = sharedInput ? selectedMedia : node.data.mediaId;
    const asset = media.find(item => item.id === mediaId);
    if (!canRunWithWeights(model, node.data, checkpointChoices(model?.id, checkpoints, jobs)) || !asset || !model.capabilities?.input_types?.includes(asset.kind)) return notify('請選擇可用模型權重與支援的輸入素材。', true);
    if (!deviceAvailable(node.data.params?.device, system)) return notify('選取的運算裝置目前不可用，請重新選擇。', true);
    if (checkpointUnavailable(node.data, checkpointChoices(model.id, checkpoints, jobs))) return notify('選取的權重已不可用，請重新選擇或檢查路徑。', true);
    try { const job = await submitJob({ kind: 'inference', model_id: model.id, media_id: mediaId, params: { ...node.data.params, ...(model.license_accepted === true ? { accept_platform_license: true } : {}) }, ...checkpointRequest(node.data) }); updateNode(id, { jobId: job.id, runMediaId: mediaId }); }
    catch (error) { notify(error.message, true); }
  }
  async function runAll() { setStartingAll(true); for (const node of nodes) await runNode(node.id); setStartingAll(false); }
  function arrange() { let offset = 28; setNodes(previous => previous.map(node => { const next = { ...node, position: { x: offset, y: 30 } }; offset += Number(node.style?.width || 420) + 32; return next; })); setTimeout(() => fitReadable(), 80); }
  function exportCanvas() {
    const blob = new Blob([JSON.stringify({ version: 1, exportedAt: new Date().toISOString(), nodes, selectedMedia, sharedInput }, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob), anchor = document.createElement('a'); anchor.href = url; anchor.download = `vision-lab-${new Date().toISOString().slice(0, 10)}.json`; anchor.click(); URL.revokeObjectURL(url); notify('已匯出畫布配置，不包含媒體與模型權重。');
  }
  async function importCanvas(event) {
    const file = event.target.files?.[0]; event.target.value = ''; if (!file) return;
    if (file.size > 2 * 1024 * 1024) return notify('配置檔不可超過 2 MB。', true);
    try { const content = JSON.parse(await file.text()); if (content.version !== 1) throw new Error('不支援的畫布版本。'); const restored = safeNodes(content.nodes, models); setNodes(restored); setSharedInput(content.sharedInput !== false); if (media.some(item => item.id === content.selectedMedia)) setSelectedMedia(content.selectedMedia); notify('畫布已匯入。請確認目前素材與本機模型狀態。'); setTimeout(() => fitReadable(restored), 80); } catch (error) { notify(`無法匯入：${error.message}`, true); }
  }
  async function onUpload(event) { const files = Array.from(event.target.files || []); event.target.value = ''; if (!files.length) return; setUploading(true); try { for (const file of files) { if (file.size > 512 * 1024 * 1024) throw new Error(`${file.name} 超過 512 MB。`); const asset = await uploadMedia(file); setSelectedMedia(asset.id); } } catch (error) { notify(error.message, true); } finally { setUploading(false); } }
  const filtered = media.filter(item => (filter === 'all' || item.kind === filter) && item.name.toLowerCase().includes(query.toLowerCase()));
  const asset = media.find(item => item.id === selectedMedia);
  const context = { models, media, jobs, system, confirmLicense, checkpoints, sharedInput, selectedMedia, updateNode, duplicateNode, removeNode, runNode, cancelJob, nodeOrder: nodes.map(node => node.id) };
  return <main className="inference-page">
    <div className="page-intro"><div><span className="eyebrow"><span/>INFERENCE PLAYGROUND</span><h1>一個畫布，多種可能<span className="accent-dot">。</span></h1><p>將模型並排，調整參數，直接比較影像與影片的辨識結果。</p></div><div className="intro-actions"><button className="button ghost" onClick={() => importInput.current?.click()}><ArrowUpFromLine size={15}/>匯入配置</button><button className="button secondary" onClick={exportCanvas}><ArrowDownToLine size={15}/>儲存畫布</button><input ref={importInput} type="file" accept=".json,application/json" hidden onChange={importCanvas}/></div></div>
    <div className="workspace-shell">
      <aside className="media-sidebar"><div className="panel-heading"><span><Layers2 size={16}/>素材庫</span><b>{media.length}</b></div><button className="upload-drop" onClick={() => fileInput.current?.click()} disabled={uploading}>{uploading ? <LoaderCircle size={22} className="spin"/> : <Upload size={22}/>}<strong>{uploading ? '正在上傳素材' : '加入你的影像或影片'}</strong><span>JPG · PNG · WEBP · MP4 · MOV</span><small>每個檔案上限 512 MB</small></button><input type="file" ref={fileInput} accept="image/jpeg,image/png,image/webp,image/gif,video/mp4,video/quicktime,video/x-msvideo,video/webm" multiple hidden onChange={onUpload}/><div className="library-heading"><span>所有素材</span><small>{filtered.length} ITEMS</small></div><div className="media-filter"><button onClick={() => setFilter('all')} className={filter === 'all' ? 'active' : ''}>全部</button><button onClick={() => setFilter('image')} className={filter === 'image' ? 'active' : ''}><Image size={12}/>圖片</button><button onClick={() => setFilter('video')} className={filter === 'video' ? 'active' : ''}><Film size={12}/>影片</button></div><label className="search-input"><Search size={14}/><input placeholder="搜尋素材名稱" value={query} onChange={event => setQuery(event.target.value)}/></label><div className="media-grid">{filtered.map(item => <button className={`media-item ${selectedMedia === item.id ? 'selected' : ''}`} key={item.id} title={item.name} onClick={() => setSelectedMedia(item.id)}><div>{item.kind === 'video' ? <><video src={item.url} preload="metadata" muted/><span className="media-kind"><Clapperboard size={15}/></span></> : <img src={item.url} alt={item.name} loading="lazy"/>}<span className="media-check">✓</span></div><span>{item.name}</span></button>)}{filtered.length === 0 && <p className="empty-text">沒有符合的素材</p>}</div><div className="media-selection"><span className="eyebrow">CURRENT INPUT</span><strong>{asset?.name || '尚未選擇素材'}</strong><span>{asset?.kind === 'video' ? '影片' : asset ? '圖片' : '從上方素材庫開始'}{asset?.width && asset?.height ? ` · ${asset.width} × ${asset.height}` : ''}</span></div></aside>
      <section className="canvas-workspace"><div className="canvas-toolbar"><div className="canvas-title"><span className="canvas-dot"/>比較畫布 <b>{nodes.length}</b></div><label className="switch-label"><input type="checkbox" checked={sharedInput} onChange={event => setSharedInput(event.target.checked)}/><span className="switch"/>共享輸入</label><div className="toolbar-separator"/><button className="button ghost small" disabled={!nodes.length || startingAll} onClick={runAll}>{startingAll ? <LoaderCircle className="spin" size={14}/> : <Play size={13}/>}全部執行</button><button className="button primary small" onClick={addNode}><Plus size={16}/>新增區塊</button></div>
        <div className="flow-wrapper"><CanvasContext.Provider value={context}><ReactFlow nodes={nodes} edges={[]} onNodesChange={onNodesChange} nodeTypes={nodeTypes} fitView minZoom={0.25} maxZoom={1.6} fitViewOptions={{ padding: 0.08, minZoom: READABLE_ZOOM, maxZoom: 1, nodes: nodes.slice(0, 2).map(node => ({ id: node.id })) }} deleteKeyCode={null} proOptions={{ hideAttribution: true }} nodesConnectable={false} panOnScroll={false} zoomOnDoubleClick={false}><Background color="#39413e" gap={22} size={1}/></ReactFlow></CanvasContext.Provider>
        {!nodes.length && <div className="empty-canvas"><span className="empty-canvas-icon"><ScanLine size={35}/></span><h2>你的第一個比較，從這裡開始</h2><p>新增區塊，選擇一個本機模型來測試。</p><button className="button primary" onClick={addNode}><Plus size={16}/>新增推論區塊</button></div>}
        <div className="canvas-bottom"><div className="canvas-controls"><button title="放大" aria-label="放大" onClick={() => flow.zoomIn({ duration: 150 })}><ZoomIn size={17}/></button><button title="縮小" aria-label="縮小" onClick={() => flow.zoomOut({ duration: 150 })}><ZoomOut size={17}/></button><i/><button title="適合畫面" aria-label="適合畫面" onClick={() => flow.fitView({ padding: 0.1, maxZoom: 1, duration: 200 })}><Expand size={17}/></button><button title="自動並排" aria-label="自動並排" onClick={arrange}><Layers2 size={17}/></button></div><span className="canvas-hint">拖曳標題移動區塊 · 滾輪縮放 · 點選邊角調整大小</span><span className="autosaved"><span/>自動儲存於此瀏覽器</span></div></div>
      </section>
    </div>
  </main>;
}
export default function InferenceCanvas(props) { return <ReactFlowProvider><CanvasInner {...props}/></ReactFlowProvider>; }
