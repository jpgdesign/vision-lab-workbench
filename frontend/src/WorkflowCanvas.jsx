import React, { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Background, Controls, Handle, MiniMap, Position, ReactFlow, ReactFlowProvider, useEdgesState, useNodesState, useReactFlow } from '@xyflow/react';
import { ArrowDownToLine, ArrowUpFromLine, Camera, Check, ChevronDown, ChevronRight, Cloud, Copy, FilePlus2, FolderOpen, GitBranch, Layers3, LoaderCircle, Maximize, Play, Plus, RefreshCw, Save, Search, Settings2, ShieldCheck, Square, Trash2, Undo2, Upload, X } from 'lucide-react';
import { activeJob, api, optionParts, statusNames, uid } from './api';
import { checkpointChoices } from './checkpoints';
import { systemDeviceOptions } from './systemDevices';
import WorkflowZoneEditor from './WorkflowZoneEditor';
import LiveCamera from './LiveCamera';
import { snapshotLiveWorkflow } from './workflowLive';
import { EMPTY_GRAPH, WORKFLOW_STORAGE_KEY, parseWorkflowGraph, workflowCatalogGroups, workflowConnectionError, workflowDefaults, workflowDraftNodeStats, workflowErrorText, workflowFromFlow, workflowGraphErrors, workflowParameterValue, workflowPortLabel, workflowRemoteModels, workflowRemoteNode, workflowTemplateProfile, workflowToFlow } from './workflowGraph';
import './WorkflowCanvas.css';

const portColors = { image: '#b7e6cc', frame: '#b7e6cc', media: '#b7e6cc', video: '#b7e6cc', detections: '#a9c9ef', tracks: '#d2b6ed', events: '#e4c18a', zones: '#d0d591', embeddings: '#efb6bf', any: '#b8c4bb' };
const safeLink = url => typeof url === 'string' && /^https?:\/\//i.test(url);
const colorFor = type => portColors[type] || '#a9c9b5';
const nodeAssets = { pose: 'yolo26n_pose', skeleton_action: 'stgcnpp_ntu120', relation: 'relateanything_convnext_large' };
const nodeTitle = (spec, node) => spec?.label || node.data.nodeType;
const emptyDraft = () => ({ name: '未命名應用流程', graph: EMPTY_GRAPH, options: { device: 'auto', frame_stride: 2, max_frames: 300 }, mediaId: '', jobId: '' });
function readDraft() {
  try {
    const saved = JSON.parse(localStorage.getItem(WORKFLOW_STORAGE_KEY));
    if (!saved) return emptyDraft();
    const defaults = emptyDraft();
    return { name: typeof saved.name === 'string' ? saved.name : defaults.name, graph: parseWorkflowGraph(saved.graph),
      mediaId: typeof saved.mediaId === 'string' ? saved.mediaId : '', jobId: typeof saved.jobId === 'string' ? saved.jobId : '',
      runContext: typeof saved.runContext?.jobId === 'string' && typeof saved.runContext?.requestKey === 'string' ? saved.runContext : null,
      options: { ...defaults.options, ...(saved.options && typeof saved.options === 'object' ? saved.options : {}) } };
  }
  catch { return emptyDraft(); }
}
function persistDraft(value) { try { localStorage.setItem(WORKFLOW_STORAGE_KEY, JSON.stringify(value)); } catch { /* Server save remains available if browser storage is full. */ } }
function exportJson(name, value) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' }));
  const anchor = document.createElement('a'); anchor.href = url; anchor.download = `${name.replace(/[\\/:*?"<>|]/g, '_') || 'workflow'}.json`; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const WorkflowNode = memo(function WorkflowNode({ id, data, selected }) {
  const { spec, stat, active, modelName } = data;
  const inputs = spec?.inputs || [], outputs = spec?.outputs || [];
  const status = stat?.status || (stat && data.jobStatus === 'completed' ? 'completed' : active ? stat ? 'running' : 'waiting' : '');
  const detail = stat?.message || stat?.summary;
  const processed = stat?.processed_frames ?? stat?.frames ?? stat?.count ?? stat?.calls;
  const elapsed = stat?.last_ms ?? stat?.elapsed_ms ?? stat?.mean_ms;
  return <article className={`wf-node ${selected ? 'is-selected' : ''} ${status ? `is-${status}` : ''}`}>
    <header><span className="wf-node-icon">{workflowRemoteNode(spec) ? <Cloud size={18}/> : <GitBranch size={18}/>}</span><div><strong>{nodeTitle(spec, { data })}</strong><small>{spec?.category || '未知節點'}</small></div>{workflowRemoteNode(spec) && <span className="wf-cloud-badge">雲端</span>}</header>
    <p>{spec?.description || '此節點尚未出現在伺服器目錄中，請檢查流程版本。'}</p>
    {modelName && <div className="wf-node-model">{modelName}{data.params.checkpoint_id && <span>微調權重</span>}</div>}
    {data.asset && <div className={`wf-node-model ${data.asset.available ? '' : 'wf-error'}`}>{data.asset.available ? '本機權重已就緒' : '本機權重尚未就緒'}</div>}
    <div className="wf-ports">{Array.from({ length: Math.max(inputs.length, outputs.length) }, (_, index) => <div className="wf-port-row" key={index}>
      <span className="wf-port-label">{inputs[index] && <><Handle type="target" position={Position.Left} id={inputs[index].name} style={{ top: '50%', background: colorFor(inputs[index].type) }} title={`${workflowPortLabel(inputs[index])} · ${inputs[index].type}`}/><span title={inputs[index].name}>{workflowPortLabel(inputs[index])}</span>{inputs[index].required && <b aria-label="必要輸入">*</b>}</>}</span>
      <span className="wf-port-label output">{outputs[index] && <><span title={outputs[index].name}>{workflowPortLabel(outputs[index])}</span><Handle type="source" position={Position.Right} id={outputs[index].name} style={{ top: '50%', background: colorFor(outputs[index].type) }} title={`${workflowPortLabel(outputs[index])} · ${outputs[index].type}`}/></>}</span>
    </div>)}</div>
    <footer><span>{status === 'waiting' ? '等待執行' : statusNames[status] || status || '待連接／設定'}</span>{Number.isFinite(processed) && <b>{processed} 幀</b>}{Number.isFinite(elapsed) && <b title={Number.isFinite(stat?.mean_ms) ? `平均 ${stat.mean_ms.toFixed(1)} ms／幀` : '本次節點處理時間'}>{elapsed.toFixed(0)} ms</b>}</footer>
    {detail && <div className="wf-node-detail" title={String(detail)}>{String(detail)}</div>}
  </article>;
});
const nodeTypes = { workflow: WorkflowNode };

function WorkflowParameter({ nodeId, param, value, modelId, models, checkpoints, jobs, remote, integration, disabled, onChange, onError }) {
  const isJson = ['json', 'object', 'array'].includes(param.type) || value !== null && typeof value === 'object';
  const [draft, setDraft] = useState(() => isJson ? JSON.stringify(value ?? param.default ?? [], null, 2) : '');
  const [error, setError] = useState('');
  useEffect(() => { if (isJson) { setDraft(JSON.stringify(value ?? param.default ?? [], null, 2)); setError(''); onError(nodeId, param.key, ''); } }, [value, nodeId, param.key]);
  const label = param.label || param.key;
  if (param.key === 'model_id' && remote) {
    const choices = workflowRemoteModels(integration);
    return <label className="field"><span>{label}</span><select value={value || ''} disabled={disabled || !choices.length} onChange={event => onChange(event.target.value)}><option value="">{choices.length ? '選擇 API 模型' : '請先完成連接設定'}</option>{value && !choices.includes(value) && <option value={value} disabled>{value} · 目前不可用</option>}{choices.map(id => <option key={id} value={id}>{id}</option>)}</select><small>{choices.length ? '使用連接服務回傳的模型清單；模型仍須支援視覺輸入。' : '請先到左側「連接」完成 API 設定，取得可用模型清單。'}</small></label>;
  }
  if (param.key === 'model_id' && (param.model_task || param.type === 'select')) {
    const allowed = (param.options || []).map(option => optionParts(option).value);
    const choices = models.filter(model => model.inference_supported && (!allowed.length || allowed.includes(model.id)) && (!param.model_task || (param.model_task === 'detection' ? ['detection', 'segmentation'].includes(model.task) : model.task === param.model_task)));
    return <label className="field"><span>{label}</span><select disabled={disabled} value={value || ''} onChange={event => onChange(event.target.value)}><option value="">選擇模型</option>{value && !choices.some(model => model.id === value) && <option value={value}>{value} · 目前不可用</option>}{choices.map(model => <option key={model.id} value={model.id}>{model.name}{!model.available ? ' · 需選擇可用權重' : ''}</option>)}</select>{param.description && <small>{param.description}</small>}</label>;
  }
  if (param.key === 'checkpoint_id') {
    const choices = checkpointChoices(modelId, checkpoints, jobs);
    return <label className="field"><span>{label}</span><select disabled={disabled || !modelId} value={value || ''} onChange={event => onChange(event.target.value)}><option value="">原始預訓練權重</option>{value && !choices.some(item => item.id === value) && <option value={value} disabled>先前權重已不可用</option>}{choices.map(item => <option key={item.id} value={item.id} disabled={!item.valid}>{item.name}{!item.valid ? ' · 路徑失效' : ''}</option>)}</select></label>;
  }
  if (param.type === 'boolean') return <label className="check-field"><input type="checkbox" checked={Boolean(value ?? param.default)} disabled={disabled} onChange={event => onChange(event.target.checked)}/><span>{label}{param.description && <small>{param.description}</small>}</span></label>;
  return <label className="field"><span>{label}</span>{isJson ? <textarea className="wf-json-editor" spellCheck="false" rows={7} value={draft} disabled={disabled} aria-invalid={Boolean(error)} onChange={event => {
    setDraft(event.target.value);
    try { const parsed = workflowParameterValue(JSON.parse(event.target.value)); setError(''); onError(nodeId, param.key, ''); onChange(parsed); }
    catch (failure) { const message = failure instanceof SyntaxError ? `${label} 的 JSON 格式不正確。` : failure.message; setError(message); onError(nodeId, param.key, message); }
  }}/> : param.type === 'select' || param.options ? <select disabled={disabled} value={value ?? param.default ?? ''} onChange={event => { const option = (param.options || []).map(optionParts).find(item => String(item.value) === event.target.value); onChange(option?.value ?? event.target.value); }}>{(param.options || []).map(option => { const item = optionParts(option); return <option key={item.value} value={item.value}>{item.label}</option>; })}</select> : ['text', 'textarea', 'multiline'].includes(param.type) && param.multiline ? <textarea rows={3} value={value ?? param.default ?? ''} disabled={disabled} onChange={event => onChange(event.target.value)}/> : <input type={['number', 'integer', 'float'].includes(param.type) ? 'number' : 'text'} disabled={disabled} value={value ?? param.default ?? ''} min={param.min} max={param.max} step={param.step || (param.type === 'integer' ? 1 : 'any')} onChange={event => onChange(['number', 'integer', 'float'].includes(param.type) && event.target.value !== '' ? Number(event.target.value) : event.target.value)}/>}{error && <small className="wf-error" role="alert">{error}</small>}{param.description && <small>{param.description}</small>}</label>;
}

function WorkflowResults({ job, jobs, onSelectJob, cancelJob }) {
  const [expanded, setExpanded] = useState(true), video = useRef(null);
  const result = job?.result, events = Array.isArray(result?.events) ? result.events : [];
  const history = jobs.filter(item => item.kind === 'workflow').slice(0, 20);
  if (!job && !history.length) return null;
  return <section className={`wf-results ${expanded ? 'expanded' : ''}`}>
    <header><button type="button" onClick={() => setExpanded(value => !value)}><ChevronDown size={15} className={!expanded ? 'wf-closed' : ''}/><strong>執行結果</strong><span>{job ? statusNames[job.status] || job.status : '選擇任務'}</span></button><select aria-label="檢視流程任務結果" value={job?.id || ''} onChange={event => onSelectJob(event.target.value)}><option value="">選擇已執行的流程</option>{history.map(item => <option key={item.id} value={item.id}>{item.workflow_name || item.name || item.id.slice(0, 8)} · {statusNames[item.status]}</option>)}</select>{activeJob(job) && <button className="button secondary small" onClick={() => cancelJob(job.id)}><Square size={12}/>停止</button>}</header>
    {expanded && <div className="wf-result-body">{result?.media_url && <div className="wf-result-media">{result.media_kind === 'video' ? <video ref={video} src={result.media_url} controls preload="metadata"/> : <img src={result.media_url} alt="流程實際執行結果"/>}</div>}<div className="wf-result-information">
      {activeJob(job) && <p className="wf-run-message"><LoaderCircle size={15} className="spin"/>{job.message || '工作已加入佇列，等待實際結果。'}{Number.isFinite(job.progress) && <b>{Math.round(job.progress * 100)}%</b>}</p>}
      {job?.status === 'failed' && <p className="wf-error" role="alert">{job.error || job.message || '流程執行失敗。'}</p>}
      {result && <><div className="wf-result-facts"><strong>{result.event_count ?? events.length} <small>筆事件</small></strong><span>{result.summary || '流程執行完成'}</span></div>{events.length ? <ol className="wf-timeline">{events.slice(0, 200).map((event, index) => { const time = Number(event.timestamp ?? event.seconds ?? event.time); return <li key={index}><button disabled={!Number.isFinite(time) || !video.current && result.media_kind !== 'video'} onClick={() => { if (video.current && Number.isFinite(time)) video.current.currentTime = Math.max(0, time); }} title="跳至影片中的事件時間">{Number.isFinite(time) ? `${time.toFixed(2)}s` : String(event.timestamp || '—')}</button><div><strong>{event.label || event.kind || '事件'}</strong><small>{[event.kind, event.track_id != null ? `追蹤 ${event.track_id}` : '', event.zone_id || event.zone, event.node_id].filter(Boolean).join(' · ')}</small></div></li>; })}</ol> : <p className="wf-hint">本次沒有符合規則的事件；可查看結果影片與節點統計。</p>}{events.length > 200 && <p className="wf-hint">畫面顯示前 200 筆；完整資料請下載事件檔案。</p>}<details className="wf-stat-details"><summary>節點執行統計</summary><pre>{JSON.stringify(result.node_stats || {}, null, 2)}</pre></details>{result.artifact_urls && <div className="wf-artifacts">{Object.entries(result.artifact_urls).map(([name, url]) => <a href={url} key={name} download><ArrowDownToLine size={13}/>{name}</a>)}</div>}</>}
      {!job && <p className="wf-hint">選擇一個流程任務查看影片、事件與節點統計。</p>}
    </div></div>}
  </section>;
}

function WorkflowCanvasInner({ models, media, jobs, checkpoints, system, notify, uploadMedia, submitWorkflow, cancelJob, confirmLicense }) {
  const initial = useRef(readDraft());
  const initialFlow = useRef(workflowToFlow(initial.current.graph));
  const [nodes, setNodes, onNodesChange] = useNodesState(initialFlow.current.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialFlow.current.edges);
  const [name, setName] = useState(initial.current.name);
  const [mediaId, setMediaId] = useState(initial.current.mediaId);
  const [options, setOptions] = useState(initial.current.options);
  const [jobId, setJobId] = useState(initial.current.jobId);
  const [runContext, setRunContext] = useState(initial.current.runContext || null);
  const [catalog, setCatalog] = useState([]), [assets, setAssets] = useState([]), [templates, setTemplates] = useState([]), [saved, setSaved] = useState([]);
  const [loading, setLoading] = useState(true), [loadError, setLoadError] = useState('');
  const [selectedId, setSelectedId] = useState(''), [panel, setPanel] = useState('templates'), [query, setQuery] = useState('');
  const [category, setCategory] = useState(''), [templateQuery, setTemplateQuery] = useState('');
  const [busy, setBusy] = useState(''), [validation, setValidation] = useState(null), [validationKey, setValidationKey] = useState('');
  const [editorErrors, setEditorErrors] = useState({}), [undo, setUndo] = useState(null);
  const [integration, setIntegration] = useState(null), [keyFile, setKeyFile] = useState(''), [integrationError, setIntegrationError] = useState('');
  const [liveWorkflow, setLiveWorkflow] = useState(null);
  const latestDraft = useRef(null);
  const importFile = useRef(null), mediaUpload = useRef(null), container = useRef(null);
  const { fitView, screenToFlowPosition } = useReactFlow();
  const job = jobs.find(item => item.id === jobId), active = activeJob(job);
  const selected = nodes.find(node => node.id === selectedId), selectedSpec = catalog.find(spec => spec.type === selected?.data.nodeType);
  const selectedMedia = media.find(item => item.id === mediaId);
  const graph = useMemo(() => workflowFromFlow(nodes, edges), [nodes, edges]);
  const requestKey = JSON.stringify({ graph, media_id: mediaId, options });
  const currentValidation = validationKey === requestKey ? validation : null;
  const stats = workflowDraftNodeStats(job, runContext, requestKey);
  const cloudNodes = nodes.filter(node => workflowRemoteNode(catalog.find(spec => spec.type === node.data.nodeType)));
  const displayNodes = useMemo(() => nodes.map(node => ({ ...node, data: { ...node.data, spec: catalog.find(spec => spec.type === node.data.nodeType), stat: stats[node.id], active, jobStatus: job?.status, asset: assets.find(asset => asset.model_id === nodeAssets[node.data.nodeType]), modelName: models.find(model => model.id === node.data.params.model_id)?.name } })), [nodes, catalog, stats, active, models, assets, job?.status]);
  const catalogGroups = useMemo(() => workflowCatalogGroups(catalog), [catalog]);
  const filteredGroups = useMemo(() => workflowCatalogGroups(catalog, query, category), [catalog, query, category]);
  const filteredCount = filteredGroups.reduce((total, group) => total + group.nodes.length, 0);
  const filteredTemplates = templates.filter(template => `${template.name} ${template.description} ${workflowTemplateProfile(template).label}`.toLocaleLowerCase().includes(templateQuery.trim().toLocaleLowerCase()));
  const setEditorError = useCallback((node, key, error) => setEditorErrors(previous => { const next = { ...previous }; if (error) next[`${node}:${key}`] = error; else delete next[`${node}:${key}`]; return next; }), []);

  const refresh = useCallback(async () => {
    setLoading(true); setLoadError('');
    const responses = await Promise.allSettled([api('/workflows/catalog'), api('/workflows/templates'), api('/workflows'), api('/workflows/integrations')]);
    if (responses[0].status === 'fulfilled') { setCatalog(responses[0].value.node_types || []); setAssets(responses[0].value.assets || []); } else setLoadError(responses[0].reason.message);
    if (responses[1].status === 'fulfilled') setTemplates(responses[1].value.templates || []); else setLoadError(previous => previous || responses[1].reason.message);
    if (responses[2].status === 'fulfilled') setSaved(responses[2].value.workflows || []);
    if (responses[3].status === 'fulfilled') setIntegration(responses[3].value.munolith || null); else setIntegrationError(responses[3].reason.message);
    setLoading(false);
  }, []);
  useEffect(() => { refresh(); }, [refresh]);
  latestDraft.current = { name, graph, mediaId, options, jobId, runContext };
  useEffect(() => { const timer = setTimeout(() => persistDraft(latestDraft.current), 300); return () => clearTimeout(timer); }, [name, graph, mediaId, options, jobId, runContext]);
  useEffect(() => { const flush = () => persistDraft(latestDraft.current); window.addEventListener('pagehide', flush); return () => { window.removeEventListener('pagehide', flush); flush(); }; }, []);
  useEffect(() => { if (!mediaId && media.length) setMediaId(media[0].id); }, [media, mediaId]);
  useEffect(() => { if (nodes.length) requestAnimationFrame(() => fitView({ padding: 0.2, duration: 0 })); }, []);
  useEffect(() => {
    const element = container.current;
    if (!element || typeof ResizeObserver === 'undefined') return;
    let scheduled, previousWidth = -1, previousHeight = -1;
    const observer = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      if (width <= 0 || height <= 0 || width === previousWidth && height === previousHeight) return;
      previousWidth = width; previousHeight = height;
      cancelAnimationFrame(scheduled);
      scheduled = requestAnimationFrame(() => fitView({ padding: 0.2, duration: 100 }));
    });
    observer.observe(element);
    return () => { observer.disconnect(); cancelAnimationFrame(scheduled); };
  }, [fitView, liveWorkflow]);

  function loadGraph(value, title, remember = true) {
    try {
      const parsed = parseWorkflowGraph(value);
      if (remember) setUndo({ name, graph });
      const flow = workflowToFlow(parsed); setNodes(flow.nodes); setEdges(flow.edges); setName(title || '匯入的應用流程'); setSelectedId(''); setEditorErrors({}); setValidation(null); setJobId('');
      setTimeout(() => fitView({ padding: 0.2, duration: 250 }), 80);
    } catch (error) { notify(error.message, true); }
  }
  function addNode(type, position) {
    if (active || nodes.length >= 48) return;
    const spec = catalog.find(item => item.type === type); if (!spec) return;
    const center = container.current?.getBoundingClientRect();
    const point = position || (center ? screenToFlowPosition({ x: center.left + center.width / 2 - 100, y: center.top + center.height / 2 - 70 }) : { x: 60 + nodes.length * 30, y: 60 + nodes.length * 30 });
    const id = uid();
    setNodes(previous => [...previous.map(node => ({ ...node, selected: false })), { id, type: 'workflow', position: point, selected: true, data: { nodeType: type, params: workflowDefaults(spec) } }]);
    setSelectedId(id); setValidation(null);
  }
  function removeNode(id) { setNodes(previous => previous.filter(node => node.id !== id)); setEdges(previous => previous.filter(edge => edge.source !== id && edge.target !== id)); setSelectedId(''); setEditorErrors(previous => Object.fromEntries(Object.entries(previous).filter(([key]) => !key.startsWith(`${id}:`)))); }
  function updateParam(key, value) {
    const hasCheckpoint = selectedSpec?.params?.some(param => param.key === 'checkpoint_id');
    const extras = key === 'model_id' && hasCheckpoint ? { checkpoint_id: '', ...(selectedSpec.params.some(param => param.key === 'use_training_classes') ? { use_training_classes: false } : {}) } : {};
    if (key === 'checkpoint_id' && selected?.data.params.model_id === 'grounding_dino_base') {
      const chosen = checkpointChoices('grounding_dino_base', checkpoints, jobs).find(item => item.id === value);
      extras.use_training_classes = chosen?.training?.caption_mode === 'chunked';
    }
    setNodes(previous => previous.map(node => node.id === selectedId ? { ...node, data: { ...node.data, params: { ...node.data.params, [key]: value, ...extras } } } : node));
  }
  function connect(connection) {
    const error = workflowConnectionError(connection, graph, catalog);
    if (error) return notify(error, true);
    setEdges(previous => [...previous, { ...connection, id: uid(), type: 'smoothstep' }]);
  }
  function errorsBeforeRun() {
    return [...workflowGraphErrors(graph, catalog), ...Object.values(editorErrors), ...(!selectedMedia ? ['請選擇存在的輸入圖片或影片。'] : []), ...(!Number.isInteger(options.frame_stride) || options.frame_stride < 1 ? ['每幾幀取樣必須是大於 0 的整數。'] : []), ...(!Number.isInteger(options.max_frames) || options.max_frames < 1 ? ['最多幀數必須是大於 0 的整數。'] : [])];
  }
  async function validateAndRun(run) {
    const errors = errorsBeforeRun(); setValidationKey(requestKey);
    if (errors.length) { setValidation({ valid: false, errors, warnings: [] }); return; }
    setBusy(run ? 'run' : 'validate');
    try {
      const payload = { graph, media_id: mediaId, options };
      const checked = await api('/workflows/validate', { method: 'POST', body: payload }); setValidation(checked); setValidationKey(requestKey);
      if (checked.valid && run) { const started = await submitWorkflow(payload); setRunContext({ jobId: started.id, requestKey }); setJobId(started.id); }
    } catch (error) { setValidation({ valid: false, errors: [error.message], warnings: [] }); }
    finally { setBusy(''); }
  }
  async function save() {
    if (!name.trim()) return notify('請先為流程命名。', true);
    if (Object.keys(editorErrors).length) return notify('請先修正節點參數的 JSON 格式。', true);
    setBusy('save');
    try { const response = await api('/workflows', { method: 'POST', body: { name: name.trim(), graph } }); const item = response.workflow || response; setSaved(previous => [item, ...previous.filter(value => value.id !== item.id)]); notify('應用流程已儲存在本機。'); }
    catch (error) { notify(error.message, true); }
    finally { setBusy(''); }
  }
  async function importGraph(event) {
    const file = event.target.files?.[0]; event.target.value = ''; if (!file) return;
    if (file.size > 500000) return notify('流程 JSON 不可超過 500 KB。', true);
    try { const document = JSON.parse(await file.text()); loadGraph(document, document.name || file.name.replace(/\.json$/i, '')); }
    catch (error) { notify(`無法匯入流程：${error.message}`, true); }
  }
  async function upload(event) {
    const file = event.target.files?.[0]; event.target.value = ''; if (!file) return;
    setBusy('upload'); try { const item = await uploadMedia(file); setMediaId(item.id); } catch (error) { notify(error.message, true); } finally { setBusy(''); }
  }
  async function configureIntegration() {
    if (!keyFile.trim()) return setIntegrationError('請輸入本機 API 金鑰檔案的完整路徑。');
    setBusy('integration'); setIntegrationError('');
    try { const response = await api('/workflows/integrations/munolith', { method: 'POST', body: { key_file_path: keyFile.trim() } }); setIntegration(response.munolith || response); setKeyFile(''); notify('雲端連接設定已儲存在本機後端。'); }
    catch (error) { setIntegrationError(error.message); }
    finally { setBusy(''); }
  }
  const locked = active || Boolean(busy);
  function openLiveWorkflow() {
    const errors = [...workflowGraphErrors(graph, catalog), ...Object.values(editorErrors)];
    if (errors.length) { setValidation({ valid: false, errors, warnings: [] }); setValidationKey(requestKey); return; }
    setLiveWorkflow(snapshotLiveWorkflow({ graph, name, options, labels: graph.nodes.map(node => catalog.find(spec => spec.type === node.type)?.label || node.type) }));
  }
  if (liveWorkflow) return <section className="wf-live-page"><div className="wf-live-navigation"><button className="button secondary small" onClick={() => setLiveWorkflow(null)}><Undo2 size={14}/>停止並返回流程畫布</button><span>流程與參數已固定，返回畫布會停止攝影機及即時任務。</span></div><LiveCamera models={models} jobs={jobs} checkpoints={checkpoints} system={system} confirmLicense={confirmLicense} workflow={liveWorkflow}/><div className="wf-attribution"><span>本機模型在本機執行；雲端節點會傳送影像。</span><a href="https://github.com/Maelic/RelateAnything" target="_blank" rel="noreferrer">Based on RelateAnything by Maëlic Neau</a></div></section>;
  return <main className="wf-page">
    <div className="wf-commandbar"><div className="wf-title"><span className="eyebrow">BUILD AN APPLICATION</span><h1>應用工作檯<span className="accent-dot">.</span></h1></div><label className="wf-name"><span className="sr-only">流程名稱</span><input value={name} disabled={locked} onChange={event => setName(event.target.value)} placeholder="為應用流程命名"/></label><div className="wf-command-actions"><button className="button secondary small" disabled={locked} onClick={save}><Save size={14}/>{busy === 'save' ? '儲存中' : '儲存'}</button><select aria-label="開啟已儲存的流程" value="" disabled={locked} onChange={event => { const item = saved.find(value => value.id === event.target.value); if (item) loadGraph(item.graph, item.name); }}><option value="">開啟流程 ({saved.length})</option>{saved.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</select><button className="icon-button" title="新增空白流程" aria-label="新增空白流程" disabled={locked} onClick={() => loadGraph(EMPTY_GRAPH, '未命名應用流程')}><FilePlus2 size={16}/></button><button className="icon-button" title="還原上一次載入前的流程" aria-label="還原上一次載入前的流程" disabled={locked || !undo} onClick={() => { loadGraph(undo.graph, undo.name, false); setUndo(null); }}><Undo2 size={16}/></button><button className="icon-button" title="匯入流程 JSON" aria-label="匯入流程 JSON" disabled={locked} onClick={() => importFile.current?.click()}><ArrowUpFromLine size={16}/></button><button className="icon-button" title="匯出流程 JSON" aria-label="匯出流程 JSON" disabled={Object.keys(editorErrors).length > 0} onClick={() => { try { exportJson(name, { name, graph: parseWorkflowGraph(graph) }); } catch (error) { notify(error.message, true); } }}><ArrowDownToLine size={16}/></button></div></div>
    <div className="wf-inputbar"><label className="field wf-source"><span>共用輸入素材</span><select value={mediaId} disabled={locked} onChange={event => setMediaId(event.target.value)}><option value="">選擇圖片或影片</option>{media.map(item => <option key={item.id} value={item.id}>{item.kind === 'video' ? '影片' : '圖片'} · {item.name}</option>)}</select></label><button className="button secondary small" disabled={locked} onClick={() => mediaUpload.current?.click()}><Upload size={14}/>上傳</button><label className="field"><span>運算裝置</span><select value={options.device} disabled={locked} onChange={event => setOptions(previous => ({ ...previous, device: event.target.value }))}>{systemDeviceOptions(system).map(item => <option key={item.value} value={item.value}>{item.label}</option>)}</select></label><label className="field wf-number"><span>每幾幀取樣</span><input type="number" min="1" max="60" value={options.frame_stride} disabled={locked} onChange={event => setOptions(previous => ({ ...previous, frame_stride: Number(event.target.value) }))}/></label><label className="field wf-number"><span>最多幀數</span><input type="number" min="1" max="3000" value={options.max_frames} disabled={locked} onChange={event => setOptions(previous => ({ ...previous, max_frames: Number(event.target.value) }))}/></label><div className="wf-run-actions"><button className="button secondary" disabled={locked || !catalog.length} onClick={openLiveWorkflow}><Camera size={14}/>即時攝影機</button><button className="button secondary" disabled={locked || !catalog.length} onClick={() => validateAndRun(false)}>{busy === 'validate' ? <LoaderCircle size={14} className="spin"/> : <ShieldCheck size={15}/>}檢查流程</button><button className="button primary" disabled={locked || !catalog.length} onClick={() => validateAndRun(true)}>{busy === 'run' ? <LoaderCircle size={14} className="spin"/> : <Play size={14} fill="currentColor"/>}執行流程</button></div></div>
    {cloudNodes.length > 0 && <div className="wf-cloud-notice"><Cloud size={14}/><span>此流程包含 {cloudNodes.length} 個雲端節點；執行時會將取樣影像送至外部 API。</span><button onClick={() => setPanel('integrations')}>連接設定 <ChevronRight size={13}/></button></div>}
    {loadError && <div className="wf-error-banner" role="alert"><span>無法讀取流程服務：{loadError}</span><button className="button ghost small" onClick={refresh}><RefreshCw size={14}/>重試</button></div>}
    <div className="wf-workspace">
      <aside className="wf-library"><div className="wf-library-tabs"><button className={panel === 'templates' ? 'active' : ''} onClick={() => setPanel('templates')}>模板</button><button className={panel === 'nodes' ? 'active' : ''} onClick={() => setPanel('nodes')}>節點</button><button className={panel === 'integrations' ? 'active' : ''} onClick={() => setPanel('integrations')}>連接</button></div><div className="wf-library-content">
        {loading ? <p className="wf-hint"><LoaderCircle size={15} className="spin"/>讀取節點目錄</p> : panel === 'nodes' ? <>
          <label className="search-input"><Search size={14}/><input aria-label="搜尋流程節點" placeholder="名稱、方法或參數…" value={query} onChange={event => setQuery(event.target.value)}/></label>
          <label className="wf-category-filter"><span>節點分類</span><select aria-label="篩選節點分類" value={category} onChange={event => setCategory(event.target.value)}><option value="">全部分類 · {catalog.length}</option>{catalogGroups.map(group => <option key={group.category} value={group.category}>{group.category} · {group.nodes.length}</option>)}</select></label>
          <p className="wf-library-count" role="status">{filteredCount} 個節點 · 點選加入或拖到畫布</p>
          {filteredGroups.map(group => <section className="wf-node-category" key={group.category}><h3>{group.category}<span>{group.nodes.length}</span></h3>{group.nodes.map(spec => <button key={spec.type} className="wf-node-library-item" title={`${spec.label}：${spec.description}`} disabled={locked} draggable={!locked} onDragStart={event => { event.dataTransfer.setData('application/vision-workflow-node', spec.type); event.dataTransfer.effectAllowed = 'copy'; }} onClick={() => addNode(spec.type)}><span><strong>{spec.label}</strong>{workflowRemoteNode(spec) ? <Cloud size={14}/> : <Plus size={14}/>}</span><small>{spec.description}</small></button>)}</section>)}
          {!filteredCount && <div className="wf-empty-small"><p>沒有符合的節點。</p><button className="button ghost small" onClick={() => { setQuery(''); setCategory(''); }}>清除篩選</button></div>}
        </> : panel === 'templates' ? <>
          <p className="wf-hint">從可編輯模板開始，串接 OpenCV、模型與規則。搜尋 OpenCV 可找到不需模型權重的影像處理示範。</p>
          <label className="search-input"><Search size={14}/><input aria-label="搜尋應用模板" placeholder="搜尋模板…" value={templateQuery} onChange={event => setTemplateQuery(event.target.value)}/></label>
          <p className="wf-library-count" role="status">{filteredTemplates.length} 個模板</p>
          {filteredTemplates.map(template => <article className="wf-template" key={template.id}><span className="wf-template-icon"><Layers3 size={18}/></span><h3>{template.name}</h3><span className="wf-template-profile">{workflowTemplateProfile(template).label}</span><p>{template.description}</p><small>{template.graph?.nodes?.length || 0} 個節點</small><button className="button secondary small full" disabled={locked} onClick={() => { loadGraph(template.graph, template.name); setPanel('nodes'); }}>使用這個模板 <ChevronRight size={13}/></button>{template.sources?.length > 0 && <details><summary>方法與來源</summary>{template.sources.filter(source => safeLink(source.url)).map(source => <a key={source.url} href={source.url} target="_blank" rel="noreferrer">{source.title} ↗</a>)}</details>}</article>)}
          {!filteredTemplates.length && <p className="wf-hint">{templates.length ? '沒有符合的模板，請更換搜尋文字。' : '尚無可用模板，可從「節點」建立流程。'}</p>}
        </> : <section className="wf-integration"><span className="wf-cloud-badge"><Cloud size={13}/>雲端 API</span><h3>雲端視覺判讀</h3><p>金鑰只由本機後端讀取。輸入已儲存 API 金鑰的檔案路徑；流程 JSON 不會包含金鑰或路徑。</p><div className={`wf-integration-state ${integration?.available ? 'ready' : ''}`}><span>{integration?.available ? '連接已就緒' : integration?.configured ? '已設定，請確認服務狀態' : '尚未設定'}</span>{integration?.model_id && <small>{integration.model_id}</small>}{integration?.base_url && <small>{integration.base_url}</small>}</div>{workflowRemoteModels(integration).length > 0 && <details className="wf-integration-models"><summary>可用 API 模型 · {workflowRemoteModels(integration).length}</summary><ul>{workflowRemoteModels(integration).map(id => <li key={id}>{id}</li>)}</ul></details>}<label className="field"><span>本機金鑰檔案路徑</span><input type="text" value={keyFile} autoComplete="off" disabled={locked} placeholder="C:\\private\\munolith-key.txt" onChange={event => setKeyFile(event.target.value)}/></label><button className="button secondary full" disabled={locked || !keyFile.trim()} onClick={configureIntegration}>設定連接</button>{(integrationError || integration?.error) && <p className="wf-error" role="alert">{integrationError || integration.error}</p>}<p className="wf-hint">雲端節點會傳送影像至此服務；其他節點使用本機模型與檔案。</p></section>}
      </div></aside>
      <section className="wf-center"><div className="wf-canvas-toolbar"><span><GitBranch size={14}/>{nodes.length} 個節點 · {edges.length} 條連線</span><span>拖曳節點 · 連接同類型接點</span><button className="icon-button" onClick={() => fitView({ padding: 0.2, duration: 250 })} title="顯示完整流程" aria-label="顯示完整流程"><Maximize size={15}/></button></div><div className="wf-flow" ref={container} onDragOver={event => { event.preventDefault(); event.dataTransfer.dropEffect = 'copy'; }} onDrop={event => { event.preventDefault(); const type = event.dataTransfer.getData('application/vision-workflow-node'); addNode(type, screenToFlowPosition({ x: event.clientX, y: event.clientY })); }}>
        <ReactFlow nodes={displayNodes} edges={edges} nodeTypes={nodeTypes} onNodesChange={active ? undefined : onNodesChange} onEdgesChange={active ? undefined : onEdgesChange} onConnect={connect} nodesDraggable={!locked} nodesConnectable={!locked} edgesReconnectable={false} deleteKeyCode={locked ? null : ['Backspace', 'Delete']} onNodeClick={(_, node) => setSelectedId(node.id)} onPaneClick={() => setSelectedId('')} onNodesDelete={deleted => { if (deleted.some(node => node.id === selectedId)) setSelectedId(''); setEditorErrors(previous => Object.fromEntries(Object.entries(previous).filter(([key]) => !deleted.some(node => key.startsWith(`${node.id}:`))))); }} defaultEdgeOptions={{ type: 'smoothstep', style: { stroke: '#81b99a', strokeWidth: 2 } }} minZoom={0.2} maxZoom={1.7} fitView colorMode="dark"><Background color="#2b3c31" gap={22} size={1}/><Controls showInteractive={false}/><MiniMap nodeColor={node => workflowRemoteNode(node.data.spec) ? '#bbaa88' : '#7ba98b'} maskColor="#101613aa" pannable zoomable/></ReactFlow>
        {!nodes.length && <div className="wf-empty-canvas"><span><GitBranch size={32} strokeWidth={1.2}/></span><h2>把影像，接成你的應用。</h2><p>選一個模板，或新增節點再連接輸入與輸出。<br/>偵測、追蹤、區域與事件都在同一張畫布。</p><button className="button secondary" onClick={() => setPanel('templates')}>瀏覽應用模板 <ChevronRight size={15}/></button></div>}
      </div>{validation && <div className={`wf-validation ${currentValidation?.valid ? 'valid' : currentValidation ? 'invalid' : ''}`} role="status"><strong>{!currentValidation ? '流程已變更，請重新檢查' : currentValidation.valid ? '流程檢查通過' : '流程需要調整'}</strong>{currentValidation && <>{(currentValidation.errors || []).map((error, i) => <p key={`e${i}`}>{workflowErrorText(error)}</p>)}{(currentValidation.warnings || []).map((warning, i) => <p className="warning" key={`w${i}`}>{workflowErrorText(warning)}</p>)}{currentValidation.valid && currentValidation.plan && <details><summary>查看執行計畫</summary><pre>{JSON.stringify(currentValidation.plan, null, 2)}</pre></details>}</>}</div>}<WorkflowResults job={job} jobs={jobs} onSelectJob={setJobId} cancelJob={cancelJob}/></section>
      <aside className="wf-inspector"><div className="wf-inspector-heading"><span><Settings2 size={15}/>節點設定</span>{selected && <div><button className="icon-button" title="複製節點" aria-label="複製節點" disabled={locked} onClick={() => { const id = uid(); setNodes(previous => [...previous.map(node => ({ ...node, selected: false })), { ...selected, id, position: { x: selected.position.x + 45, y: selected.position.y + 45 }, selected: true, data: structuredClone(selected.data) }]); setSelectedId(id); }}><Copy size={14}/></button><button className="icon-button danger-hover" title="刪除節點" aria-label="刪除節點" disabled={locked} onClick={() => removeNode(selected.id)}><Trash2 size={14}/></button></div>}</div>{selected ? <div className="wf-inspector-content"><span className="eyebrow">{selectedSpec?.category || 'NODE'}</span><h2>{selectedSpec?.label || selected.data.nodeType}</h2><p className="wf-hint">{selectedSpec?.description}</p>{selected.data.nodeType.startsWith('cv_') && <div className="wf-opencv-help"><strong>OpenCV 處理節點</strong>{selectedSpec?.outputs?.some(port => port.name === 'frame') ? <p>把「影像」輸出接到下一個處理／模型節點。連到「畫面與事件輸出」可查看處理後的影像；該輸出改接原始來源，就能把結果畫回原圖。</p> : <p>把「追蹤物件」接到區域規則，把「動作／狀態」接到條件節點，即可組合移動與停留事件。</p>}{['cv_background', 'cv_optical_flow', 'cv_track_motion', 'cv_histogram'].includes(selected.data.nodeType) && <p>使用影片或即時來源取得連續觀察。這裡的移動／外觀變化是數值規則，可接條件節點，不代表已辨識特定行為。</p>}{selected.data.nodeType === 'cv_contours' && <p>輪廓的「偵測框」可接追蹤與區域規則；標籤由你設定，並非模型辨識出的物件類別。</p>}</div>}{workflowRemoteNode(selectedSpec) && <p className="wf-remote-description"><Cloud size={14}/>此節點會將影像傳送至雲端 API。</p>}{selectedSpec?.params?.some(param => param.key === 'zones') && <WorkflowZoneEditor key={selected.id} media={selectedMedia} zones={selected.data.params.zones} onChange={value => updateParam('zones', value)} disabled={locked}/>}<div className="wf-parameter-fields">{(selectedSpec?.params || []).map(param => <WorkflowParameter key={`${selected.id}:${param.key}`} nodeId={selected.id} param={param} value={selected.data.params[param.key]} modelId={selected.data.params.model_id} remote={workflowRemoteNode(selectedSpec)} integration={integration} models={models} checkpoints={checkpoints} jobs={jobs} disabled={locked} onChange={value => updateParam(param.key, value)} onError={setEditorError}/>)}</div>{!selectedSpec?.params?.length && <p className="wf-hint">此節點使用共用輸入或上游結果，沒有額外參數。</p>}<details className="wf-port-info"><summary>接點與資料型別</summary>{[...(selectedSpec?.inputs || []).map(item => ({ ...item, direction: '輸入' })), ...(selectedSpec?.outputs || []).map(item => ({ ...item, direction: '輸出' }))].map((item, i) => <p key={i}><b>{item.direction} · {workflowPortLabel(item)}</b><span style={{ color: colorFor(item.type) }}>{String(item.type)}</span></p>)}</details><small className="wf-node-id">{selected.id}</small></div> : <div className="wf-inspector-empty"><MouseHint/><h3>選取節點，調整應用</h3><p>點擊畫布上的節點來選模型、設定規則，或直接在素材上繪製區域。</p></div>}</aside>
    </div>
    <div className="wf-attribution"><span>流程草稿自動保存在此瀏覽器 · 刪除連線：選取後按 Delete</span><a href="https://github.com/Maelic/RelateAnything" target="_blank" rel="noreferrer">Based on RelateAnything by Maëlic Neau</a></div>
    <input ref={importFile} type="file" accept=".json,application/json" hidden onChange={importGraph}/><input ref={mediaUpload} type="file" accept="image/*,video/*" hidden onChange={upload}/>
  </main>;
}
function MouseHint() { return <Settings2 size={29} strokeWidth={1.2}/>; }
export default function WorkflowCanvas(props) { return <ReactFlowProvider><WorkflowCanvasInner {...props}/></ReactFlowProvider>; }
