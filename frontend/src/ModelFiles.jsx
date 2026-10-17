import React, { useEffect, useRef, useState } from 'react';
import { Cpu, FolderSearch, Link2, LoaderCircle, Plus, Trash2, Unlink } from 'lucide-react';
import { api, byteSize } from './api';
import './modelFiles.css';

const capacity = value => Number.isFinite(value) && value >= 0 ? `${(value / 1024 ** 3).toFixed(1)} GB` : '—';

export function HardwarePanel({ system }) {
  const usableGpus = (system?.devices || []).filter(device => /^cuda:\d+$/.test(device.id));
  return <section className="workspace-hardware library-section">
    <div className="content-section-title"><div><h2>運算環境</h2><p>裝置選項依目前 Python／PyTorch 可用的硬體提供。</p></div><Cpu size={19}/></div>
    <div className="hardware-grid">
      <div><span>作業系統 / Python</span><strong>{system ? `${system.platform?.system || '—'} · Python ${system.python?.version || '—'}` : '讀取中'}</strong><small>{system?.cpu?.logical_count ?? '—'} 個邏輯核心</small></div>
      <div><span>記憶體 RAM</span><strong>{capacity(system?.memory?.total_bytes)}</strong><small>目前可用 {capacity(system?.memory?.available_bytes)}</small></div>
      <div><span>可用 CUDA 裝置</span><strong>{usableGpus.length ? `${usableGpus.length} 張 GPU` : 'CPU 模式'}</strong><small>{system?.torch?.version ? `PyTorch ${system.torch.version}${system.torch.cuda_build ? ` · CUDA ${system.torch.cuda_build}` : ''}` : '尚未安裝 PyTorch'}</small></div>
      <div><span>工作空間磁碟</span><strong>{capacity(system?.disk?.free_bytes)} 可用</strong><small>總容量 {capacity(system?.disk?.total_bytes)}</small></div>
    </div>
    {usableGpus.map(device => <div className="hardware-device" key={device.id}><span><Cpu size={14}/><strong>{device.name}</strong><code>{device.id}</code></span><span>VRAM {capacity(device.memory_total_bytes)}</span></div>)}
    {system?.nvidia?.detected && !system?.torch?.cuda_available && <p className="workspace-notice">偵測到 NVIDIA GPU，但目前 PyTorch 尚無法使用 CUDA。完成相容的 PyTorch／驅動安裝後，重新檢查環境。</p>}
    {system?.torch?.error && <p className="workspace-notice">PyTorch 環境檢查：{system.torch.error}</p>}
    {system?.python?.supported === false && <p className="workspace-notice">目前 Python 版本不在支援範圍；建議使用 Python {system.python.recommended || '3.11'} 建立環境。</p>}
    {system?.recommendations?.notes?.length > 0 && <details className="hardware-notes"><summary>環境建議</summary>{system.recommendations.notes.map((note, index) => <p key={index}>{note}</p>)}</details>}
  </section>;
}

export default function ModelFiles({ models, checkpoints = [], refresh, notify }) {
  const [snapshot, setSnapshot] = useState(null), [roots, setRoots] = useState(''), [busy, setBusy] = useState(''), [error, setError] = useState('');
  const [baseModel, setBaseModel] = useState(models[0]?.id || ''), [basePath, setBasePath] = useState('');
  const [checkpointModel, setCheckpointModel] = useState(models[0]?.id || ''), [checkpointPath, setCheckpointPath] = useState(''), [checkpointName, setCheckpointName] = useState('');
  const alive = useRef(true);
  useEffect(() => {
    alive.current = true; const controller = new AbortController();
    api('/model-files', { signal: controller.signal }).then(value => { if (alive.current) setSnapshot(value); }).catch(failure => { if (alive.current && failure.name !== 'AbortError') setError(failure.message); });
    return () => { alive.current = false; controller.abort(); };
  }, []);
  async function action(key, request, success, refreshModels = false) {
    if (busy) return;
    setBusy(key); setError('');
    try {
      const value = await request();
      if (!alive.current) return;
      setSnapshot(value?.roots ? value : await api('/model-files'));
      if (refreshModels) await refresh();
      if (alive.current) notify(success);
    } catch (failure) { if (alive.current) { setError(failure.message); notify(failure.message, true); } }
    finally { if (alive.current) setBusy(''); }
  }
  const scan = () => action('scan', () => api('/model-files/scan', { method: 'POST', body: { roots: [...new Set(roots.split(/\r?\n/).map(value => value.trim()).filter(Boolean))] } }), '模型資料夾掃描完成。', true);
  const bind = (modelId, path) => action(`bind:${modelId}:${path}`, () => api('/model-files/bind', { method: 'POST', body: { model_id: modelId, path } }), '已綁定本機原始權重。', true);
  const register = (modelId, path, name = '') => action(`register:${modelId}:${path}`, () => api('/checkpoints', { method: 'POST', body: { model_id: modelId, path, ...(name.trim() ? { name: name.trim() } : {}) } }), '微調權重已加入推論與即時攝影機的選單。', true);
  const external = checkpoints.filter(item => item.source === 'external');
  const modelName = id => models.find(model => model.id === id)?.name || id;
  return <section className="model-files library-section">
    <div className="content-section-title"><div><h2>接入本機權重</h2><p>使用已下載的原始模型與訓練成果，直接從原路徑讀取。</p></div><span className="tiny-label">MODEL PATHS</span></div>
    {error && <p className="workspace-error" role="alert">{error}</p>}
    <div className="model-files-grid">
      <div className="model-file-form"><h3><FolderSearch size={16}/>掃描模型資料夾</h3><label className="field"><span>額外資料夾 · 每行一個絕對路徑</span><textarea value={roots} disabled={Boolean(busy)} onChange={event => setRoots(event.target.value)} rows={3} placeholder={'D:\\models\n/home/user/models'}/></label><p>留空時沿用目前設定與預設資料夾。若找到多份相容權重，請選擇一份綁定。</p><button className="button secondary" disabled={Boolean(busy)} onClick={scan}>{busy === 'scan' ? <LoaderCircle size={14} className="spin"/> : <FolderSearch size={14}/>} {busy === 'scan' ? '正在掃描指定資料夾' : '掃描權重'}</button><details className="model-roots"><summary>目前搜尋位置 · {snapshot?.roots?.length || 0}</summary>{snapshot?.roots?.map((root, index) => <div key={`${root.path}:${index}`}><code>{root.path}</code><small>{root.exists === false ? '資料夾不存在' : root.source || '搜尋位置'}</small></div>)}</details></div>
      <div className="model-file-form"><h3><Link2 size={16}/>指定原始模型權重</h3><label className="field"><span>模型架構</span><select value={baseModel} disabled={Boolean(busy)} onChange={event => setBaseModel(event.target.value)}>{models.map(model => <option value={model.id} key={model.id}>{model.name}</option>)}</select></label><label className="field"><span>權重檔案或完整模型資料夾</span><input value={basePath} disabled={Boolean(busy)} onChange={event => setBasePath(event.target.value)} placeholder="貼上此電腦上的絕對路徑"/></label><p>Hugging Face 模型請選包含設定與處理器的完整資料夾；單檔模型可指定 .pt／.pth 權重。</p><button className="button primary" disabled={Boolean(busy) || !baseModel || !basePath.trim()} onClick={() => bind(baseModel, basePath.trim())}><Link2 size={14}/>驗證並綁定</button></div>
    </div>
    {snapshot?.bindings?.length > 0 && <div className="model-path-list"><h3>自訂原始權重</h3>{snapshot.bindings.map(binding => <div className="model-path-row" key={binding.model_id}><div><strong>{modelName(binding.model_id)} <span className={binding.valid ? 'path-valid' : 'path-invalid'}>{binding.valid ? '可用' : '路徑失效'}</span></strong><code>{binding.path}</code>{binding.error && <p>{binding.error}</p>}</div><button className="button ghost small" disabled={Boolean(busy)} title="只解除綁定，保留原始檔案" onClick={() => action(`unbind:${binding.model_id}`, () => api(`/model-files/bind/${encodeURIComponent(binding.model_id)}`, { method: 'DELETE' }), '已解除綁定，原始檔案保留。', true)}><Unlink size={13}/>解除綁定</button></div>)}</div>}
    {snapshot?.scan_errors?.length > 0 && <div className="workspace-notice">{snapshot.scan_errors.map((item, index) => <p key={index}>{typeof item === 'string' ? item : `${item.path || ''} ${item.error || item.message || ''}`}</p>)}</div>}
    {snapshot?.truncated && <p className="workspace-notice">掃描已達上限，請指定範圍較小的模型資料夾。</p>}
    <div className="model-path-list"><h3>掃描結果 <small>{snapshot?.candidates?.length || 0}</small></h3>{snapshot?.candidates?.length ? snapshot.candidates.map((candidate, index) => <div className="model-path-row" key={`${candidate.path}:${candidate.model_id}:${index}`}><div><strong>{modelName(candidate.model_id)} <span className={candidate.valid ? 'path-valid' : 'path-invalid'}>{candidate.kind === 'checkpoint' ? '微調權重' : '原始權重'} · {candidate.valid ? '可用' : '未通過檢查'}</span></strong><code>{candidate.path}</code><small>{byteSize(candidate.bytes)}</small>{candidate.error && <p>{candidate.error}</p>}</div><button className="button secondary small" disabled={Boolean(busy) || !candidate.valid || !candidate.model_id} onClick={() => candidate.kind === 'checkpoint' ? register(candidate.model_id, candidate.path) : bind(candidate.model_id, candidate.path)}>{candidate.kind === 'checkpoint' ? <Plus size={13}/> : <Link2 size={13}/>} {candidate.kind === 'checkpoint' ? '登記微調權重' : '綁定原始權重'}</button></div>) : <p className="model-path-empty">先掃描模型資料夾，或在上方直接指定權重路徑。</p>}</div>
    <div className="checkpoint-register"><div><h3>加入外部微調權重</h3><p>登記後可在推論與即時攝影機的「模型權重」選取。模型設定、類別資訊需與權重一起保留。</p></div><div className="checkpoint-register-fields"><label className="field"><span>模型架構</span><select value={checkpointModel} disabled={Boolean(busy)} onChange={event => setCheckpointModel(event.target.value)}>{models.filter(model => model.inference_supported).map(model => <option value={model.id} key={model.id}>{model.name}</option>)}</select></label><label className="field"><span>微調權重絕對路徑</span><input value={checkpointPath} disabled={Boolean(busy)} onChange={event => setCheckpointPath(event.target.value)} placeholder="檔案或完整模型資料夾"/></label><label className="field"><span>顯示名稱（選填）</span><input value={checkpointName} maxLength={120} disabled={Boolean(busy)} onChange={event => setCheckpointName(event.target.value)} placeholder="例如：工廠零件辨識 v2"/></label><button className="button primary" disabled={Boolean(busy) || !checkpointModel || !checkpointPath.trim()} onClick={() => register(checkpointModel, checkpointPath.trim(), checkpointName)}><Plus size={14}/>驗證並登記</button></div><p className="checkpoint-format-note">YOLO 微調權重需保留工作台產生的類別／訓練設定檔；其他架構也需保留對應設定。若格式不完整，驗證會指出缺少項目。</p></div>
    {external.length > 0 && <div className="model-path-list"><h3>已登記外部微調權重</h3>{external.map(item => <div className="model-path-row" key={item.id}><div><strong>{item.name || item.id} <span className={item.valid === false ? 'path-invalid' : 'path-valid'}>{item.valid === false ? '路徑失效' : '可用'} · {modelName(item.model_id)}</span></strong><code>{item.path}</code>{item.error && <p>{item.error}</p>}</div><button className="button ghost small" disabled={Boolean(busy)} title="只移除清單登記，保留權重檔案" onClick={() => action(`remove:${item.id}`, () => api(`/checkpoints/${encodeURIComponent(item.id)}`, { method: 'DELETE' }), '已移除登記，權重檔案保留。', true)}><Trash2 size={13}/>移除登記</button></div>)}</div>}
  </section>;
}
