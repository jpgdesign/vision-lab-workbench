import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Aperture, Boxes, Camera, Check, CircleHelp, Cpu, LayoutDashboard, LoaderCircle, RefreshCw, ScanLine, Settings2, X } from 'lucide-react';
import { api, activeJob } from './api';
import InferenceCanvas from './InferenceCanvas';
import Training from './Training';
import ModelLibrary from './ModelLibrary';
import LiveCamera from './LiveCamera';
import { applySystemDevices } from './systemDevices';

export default function App() {
  const [page, setPage] = useState('inference');
  const [models, setModels] = useState([]);
  const [media, setMedia] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [health, setHealth] = useState(null);
  const [system, setSystem] = useState(null), [checkpoints, setCheckpoints] = useState([]);
  const [connectionError, setConnectionError] = useState('');
  const [loading, setLoading] = useState(true);
  const [toast, setToast] = useState(null);
  const [help, setHelp] = useState(false);
  const toastTimer = useRef(null);
  const notify = useCallback((message, error = false) => {
    clearTimeout(toastTimer.current); setToast({ message, error });
    toastTimer.current = setTimeout(() => setToast(null), error ? 10000 : 4500);
  }, []);
  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const values = await Promise.all([api('/health'), api('/models'), api('/media'), api('/jobs'), api('/datasets'), api('/system'), api('/checkpoints')]);
      setHealth(values[0]); setModels(applySystemDevices(values[1].models || [], values[5])); setMedia(values[2].media || []); setJobs(values[3].jobs || []); setDatasets(values[4].datasets || []); setSystem(values[5]); setCheckpoints(values[6].checkpoints || []); setConnectionError('');
    } catch (error) { setConnectionError(error.message); }
    finally { setLoading(false); }
  }, []);
  useEffect(() => { refresh(); return () => clearTimeout(toastTimer.current); }, [refresh]);
  useEffect(() => {
    let alive = true, timer;
    async function poll() {
      try {
        const [jobData, healthData, checkpointData] = await Promise.all([api('/jobs'), api('/health'), api('/checkpoints')]);
        if (alive) { setJobs(jobData.jobs || []); setHealth(healthData); setCheckpoints(checkpointData.checkpoints || []); setConnectionError(''); }
      } catch (error) { if (alive) setConnectionError(error.message); }
      if (alive) timer = setTimeout(poll, 2200);
    }
    timer = setTimeout(poll, 2200);
    return () => { alive = false; clearTimeout(timer); };
  }, []);
  async function submitJob(payload) {
    const job = await api('/jobs', { method: 'POST', body: payload });
    const value = job.job || job;
    setJobs(prev => [value, ...prev.filter(item => item.id !== value.id)]);
    notify('任務已加入佇列，會依序使用本機運算資源。');
    return value;
  }
  async function cancelJob(id) {
    try { const data = await api(`/jobs/${encodeURIComponent(id)}/cancel`, { method: 'POST' }); const job = data.job || data; setJobs(prev => prev.map(item => item.id === id ? job : item)); notify('已送出取消要求。'); }
    catch (error) { notify(error.message, true); }
  }
  async function uploadMedia(file) {
    const form = new FormData(); form.append('file', file);
    const data = await api('/media', { method: 'POST', body: form }); const item = data.media || data;
    setMedia(prev => [item, ...prev.filter(asset => asset.id !== item.id)]);
    notify(`已加入 ${item.name}`); return item;
  }
  async function confirmLicense(modelId) {
    try {
      await api(`/model-licenses/${encodeURIComponent(modelId)}`, { method: 'POST', body: { accepted: true } });
      const result = await api('/models'); setModels(applySystemDevices(result.models || [], system));
      notify('已記住此模型的授權確認。');
    } catch (error) { notify(error.message, true); }
  }
  const runningCount = jobs.filter(activeJob).length;
  const usableGpu = system?.torch?.cuda_available && system?.torch?.working !== false ? system.devices?.find(device => /^cuda:\d+$/.test(device.id)) : null;
  const environmentLabel = !system ? '環境檢查中' : usableGpu ? usableGpu.name : system.nvidia?.detected ? '偵測到 GPU · CUDA 未就緒' : !system.torch?.installed ? 'CPU · 尚未安裝 PyTorch' : system.torch?.working === false ? 'CPU · PyTorch 環境待修復' : 'CPU 模式';
  const shared = { models, media, jobs, datasets, setDatasets, system, checkpoints, notify, submitJob, cancelJob, uploadMedia, confirmLicense };
  return <div className="app-shell">
    <header className="app-header">
      <button className="brand" onClick={() => setPage('inference')} aria-label="Vision Lab 首頁"><span className="brand-mark"><Aperture size={25} strokeWidth={1.8} /></span><span>VISION<span className="brand-light">LAB</span><small>視覺研究工作台</small></span></button>
      <nav className="top-nav" aria-label="工作區">
        <button className={page === 'inference' ? 'active' : ''} onClick={() => setPage('inference')}><ScanLine size={17} />推論比較</button>
        <button className={page === 'live' ? 'active' : ''} onClick={() => setPage('live')}><Camera size={17} />即時攝影機</button>
        <button className={page === 'training' ? 'active' : ''} onClick={() => setPage('training')}><Settings2 size={17} />微調訓練</button>
        <button className={page === 'models' ? 'active' : ''} onClick={() => setPage('models')}><Boxes size={17} />模型與任務{runningCount > 0 && <b className="nav-count">{runningCount}</b>}</button>
      </nav>
      <div className="header-end"><span className={`local-state ${connectionError ? 'offline' : ''}`}><i />{connectionError ? '連線中斷' : 'LOCAL WORKSPACE'}</span><button className="icon-button" title="操作說明" aria-label="操作說明" onClick={() => setHelp(true)}><CircleHelp size={19} /></button></div>
    </header>
    {connectionError && <div className="connection-banner"><span>無法連線到本機服務：{connectionError}</span><button onClick={refresh}><RefreshCw size={14} />重新連線</button></div>}
    {loading && !models.length ? <div className="loading-page"><LoaderCircle className="spin" size={26}/><h2>正在連接研究工作台</h2><p>讀取本機模型、教學素材與任務紀錄</p>{connectionError && <button className="button secondary" onClick={refresh}>重試</button>}</div> : <>
      <div style={{ display: page === 'inference' ? 'contents' : 'none' }}><InferenceCanvas {...shared} visible={page === 'inference'} /></div>
      {page === 'live' && <LiveCamera {...shared} />}
      {page === 'training' && <Training {...shared} />}
      {page === 'models' && <ModelLibrary {...shared} refresh={refresh} />}
    </>}
    <footer className="app-status"><span><span className="status-dot" />所有檔案與運算保留在本機</span><div><span><Cpu size={12} />{environmentLabel}</span><span>{usableGpu?.memory_total_bytes ? `${(usableGpu.memory_total_bytes / 1024 ** 3).toFixed(0)} GB VRAM` : ''}</span><span>{runningCount ? `${runningCount} 個任務處理中` : '佇列閒置'}</span></div><span>VISION LAB <b>v1.0</b></span></footer>
    {toast && <div className={`toast ${toast.error ? 'error' : ''}`} role="status">{toast.error ? <CircleHelp size={18} /> : <Check size={18}/>}<span>{toast.message}</span><button className="icon-button" onClick={() => setToast(null)} aria-label="關閉訊息"><X size={15}/></button></div>}
    {help && <div className="modal-backdrop" onClick={() => setHelp(false)}><section className="modal" role="dialog" aria-modal="true" aria-labelledby="help-title" onClick={event => event.stopPropagation()}><button className="icon-button modal-close" onClick={() => setHelp(false)} aria-label="關閉說明"><X size={20}/></button><span className="eyebrow">GETTING STARTED</span><h2 id="help-title">從一張影像，開始比較。</h2><div className="help-step"><b>01</b><div><h3>選擇素材，新增比較區塊</h3><p>左側選取教學圖片或上傳影片。開啟「共享輸入」即可讓每個區塊比較同一份素材。</p></div></div><div className="help-step"><b>02</b><div><h3>選模型、調參數，按下開始</h3><p>每張卡片獨立設定。拖動卡片標題可排列，選取後拖動邊角可調整大小；滑鼠滾輪縮放畫布。</p></div></div><div className="help-step"><b>03</b><div><h3>驗證資料集，再啟動微調</h3><p>依模型選取支援格式，使用本機資料夾或 ZIP。只有通過驗證的資料集能開始訓練；模型支援範圍與限制會直接顯示。</p></div></div><div className="note-box">所有推論與訓練共用單一序列佇列。畫布上的數據只來自實際執行的任務；尚未完成的項目不會顯示模擬結果。</div><button className="button primary full" onClick={() => setHelp(false)}>開始使用 <LayoutDashboard size={16}/></button></section></div>}
  </div>;
}
