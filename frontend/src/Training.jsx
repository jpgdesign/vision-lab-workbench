import React, { useEffect, useMemo, useRef, useState } from 'react';
import { ArrowDownToLine, ArrowRight, Check, ChevronDown, CircleAlert, Database, FileArchive, FileCheck2, FileText, FolderOpen, Gauge, LoaderCircle, Play, RotateCcw, Settings2, Square, Terminal, Upload, Waypoints } from 'lucide-react';
import { api, activeJob, optionParts, statusNames, taskNames } from './api';
import { ParameterField } from './InferenceCanvas';
import { deviceAvailable } from './systemDevices';

const formatNames = { yolo_detection: 'YOLO Detection', coco_detection: 'COCO Detection', imagefolder: 'ImageFolder', video_csv: 'Video CSV' };
const parameterInfo = {
  epochs: ['訓練輪數', 1, 10000, 1], batch_size: ['批次大小', 1, 1024, 1], imgsz: ['輸入尺寸', 32, 4096, 16], lr: ['學習率', 0.00000001, 1, 0.0001], weight_decay: ['權重衰減', 0, 1, 0.0001], freeze: ['凍結層數', 0, 1000, 1], patience: ['提前停止耐心值', 0, 10000, 1], seed: ['隨機種子', 0, 4294967295, 1], workers: ['資料載入程序', 0, 32, 1], accumulate: ['梯度累積步數', 1, 1024, 1], label_smoothing: ['標籤平滑', 0, 1, 0.01], gradient_clip: ['梯度裁切', 0, 100, 0.1], momentum: ['動量', 0, 1, 0.01], warmup_epochs: ['暖身輪數', 0, 1000, 1], box: ['框回歸損失權重', 0, 100, 0.1], cls: ['分類損失權重', 0, 100, 0.1], dfl: ['DFL 損失權重', 0, 100, 0.1], num_frames: ['每段影格數', 1, 128, 1], frame_stride: ['影格取樣間距', 1, 64, 1], step_size: ['衰減間隔', 1, 1000, 1], gamma: ['衰減倍率', 0, 1, 0.01], focal_gamma: ['Focal γ', 0, 10, 0.1], max_batches: ['每輪最大批次', 0, 100000, 1], max_train_batches: ['每輪最多訓練批次', 0, 100000, 1], max_val_batches: ['每輪最多驗證批次', 0, 100000, 1], dropout: ['Dropout', 0, 1, 0.01], max_grad_norm: ['梯度裁切', 0, 100, 0.1], val_every: ['驗證間隔', 1, 10000, 1]
};
const stringLabels = { device: '運算裝置', optimizer: '最佳化器', loss: '損失函數', scheduler: '學習率排程', strategy: '微調策略', augmentation: '資料增強', amp: '混合精度 AMP', cos_lr: '餘弦學習率', deterministic: '確定性模式', pretrained: '載入預訓練權重' };
const choiceNames = { full: '全部參數微調', head_only: '只訓練分類頭', freeze_backbone: '凍結骨幹網路', cosine: 'Cosine · 餘弦衰減', linear: 'Linear · 線性衰減', step: 'Step · 階梯衰減', none: 'None · 固定', cross_entropy: 'Cross Entropy', ce: 'Cross Entropy', focal: 'Focal Loss', yolo_composite: 'YOLO 原生複合損失', rf_detr_composite: 'RF-DETR 原生複合損失', grounding_dino_composite: 'Grounding DINO 原生複合損失', auto: '自動選擇' };

function LossChart({ metrics = [] }) {
  const valid = metrics.filter(metric => Number.isFinite(Number(metric.train_loss)) && metric.train_loss != null);
  if (!valid.length) return <div className="empty-chart"><Gauge size={28} strokeWidth={1.2}/><strong>等待第一輪訓練數據</strong><span>開始訓練後，這裡會顯示實際 loss 曲線</span></div>;
  const values = valid.flatMap(metric => [metric.train_loss, metric.val_loss].filter(value => value != null && Number.isFinite(Number(value))).map(Number));
  const min = Math.max(0, Math.min(...values) * 0.9), max = Math.max(...values) * 1.1 || 1;
  const left = 42, right = 375, top = 18, bottom = 162;
  const x = index => valid.length === 1 ? left + (right - left) / 2 : left + index / (valid.length - 1) * (right - left);
  const y = value => bottom - (Number(value) - min) / Math.max(max - min, 0.000001) * (bottom - top);
  return <div className="loss-chart"><div className="chart-legend"><span><i/>Train loss</span><span><i className="val"/>Val loss</span></div><svg viewBox="0 0 400 196" role="img" aria-label={`訓練損失折線圖，共 ${valid.length} 筆紀錄`}>
    {[0, 1, 2, 3].map(index => { const value = min + (max - min) * index / 3, height = y(value); return <g key={index}><line x1={left} x2={right} y1={height} y2={height} stroke="#2d3732" strokeDasharray="3 5"/><text x={left - 9} y={height + 4} textAnchor="end">{value.toFixed(2)}</text></g>; })}
    <polyline fill="none" stroke="#a8e1c0" strokeWidth="2.4" strokeLinejoin="round" points={valid.map((metric, index) => `${x(index)},${y(metric.train_loss)}`).join(' ')}/>
    <polyline fill="none" stroke="#b7a0ec" strokeWidth="2" strokeLinejoin="round" points={valid.map((metric, index) => metric.val_loss != null && Number.isFinite(Number(metric.val_loss)) ? `${x(index)},${y(metric.val_loss)}` : '').filter(Boolean).join(' ')}/>
    {valid.length === 1 && <circle cx={x(0)} cy={y(valid[0].train_loss)} r="4" fill="#a8e1c0"/>}
    {valid.map((metric, index) => (index === 0 || index === valid.length - 1 || (valid.length < 8)) && <text key={index} x={x(index)} y="185" textAnchor="middle">{metric.epoch ?? index + 1}</text>)}<text x="208" y="195" textAnchor="middle">Epoch</text>
  </svg></div>;
}
export function JobMonitor({ job, cancelJob, models, compact = false }) {
  const metrics = job?.metrics || [], latest = metrics.at(-1);
  const validationMap = latest?.map50_95 ?? latest?.map;
  const result = job?.result;
  const completionSummary = typeof result?.summary === 'string' ? result.summary : '';
  const stoppingReason = result?.training?.early_stopping?.reason || result?.early_stopping?.reason;
  const model = models.find(model => model.id === job?.model_id);
  return <div className={`job-monitor ${compact ? 'compact' : ''}`}>
    <div className="monitor-header"><div><span className="eyebrow">LIVE MONITOR</span><h3>訓練觀測</h3></div>{job ? <span className={`status-pill ${job.status}`}><i/>{statusNames[job.status]}</span> : <span className="status-pill idle">尚未開始</span>}</div>
    <div className="monitor-session"><span>目前實驗</span><strong>{model?.name || '等待啟動訓練'}</strong><small>{job ? `ID · ${job.id.slice(0, 12)}` : '通過資料驗證後，即可建立第一個實驗。'}</small></div>
    {job && <div className="progress-block"><div><span>{job.message || statusNames[job.status]}</span><b>{Math.round(Math.max(0, Math.min(1, job.progress || 0)) * 100)}%</b></div><div className="progress-track"><span style={{ width: `${Math.max(0, Math.min(1, job.progress || 0)) * 100}%` }}/></div></div>}
    <div className="monitor-metrics"><div><span>TRAIN LOSS</span><strong>{latest?.train_loss != null ? Number(latest.train_loss).toFixed(4) : '—'}</strong></div><div><span>VAL LOSS</span><strong>{latest?.val_loss != null ? Number(latest.val_loss).toFixed(4) : '—'}</strong></div><div><span>EPOCH</span><strong>{latest?.epoch ?? '—'}{job?.params?.epochs && <small> / {job.params.epochs}</small>}</strong></div></div>
    <LossChart metrics={metrics}/>
    {latest?.accuracy != null && <div className="additional-metric">驗證 Accuracy <b>{(Number(latest.accuracy) * 100).toFixed(2)}%</b></div>}
    {latest?.map50 != null && <div className="additional-metric">驗證 mAP@50 <b>{(Number(latest.map50) * 100).toFixed(2)}%</b></div>}
    {validationMap != null && Number.isFinite(Number(validationMap)) && <div className="additional-metric">驗證 mAP50–95 <b>{(Number(validationMap) * 100).toFixed(2)}%</b></div>}
    {job?.status === 'completed' && (completionSummary || stoppingReason) && <section className={`training-outcome ${stoppingReason ? 'early-stopped' : ''}`}><strong>{stoppingReason ? 'Early Stopping · 已提前停止' : '訓練完成'}</strong>{completionSummary && <p>{completionSummary}</p>}{typeof stoppingReason === 'string' && <p className="stop-reason">{stoppingReason}</p>}</section>}
    {job?.error && <div className="inline-error monitor-error">{job.error}</div>}
    <details className="log-panel" open={job?.status === 'failed'}><summary><Terminal size={14}/>執行紀錄 <span>{job?.logs?.length || 0}</span><ChevronDown size={14}/></summary><pre>{job?.logs?.length ? job.logs.join('\n') : '尚無執行紀錄。啟動後會顯示實際訓練輸出。'}</pre></details>
    {activeJob(job) && <button className="button danger full" onClick={() => cancelJob(job.id)}><Square size={13}/>取消訓練</button>}
    {result?.checkpoint_url && <a className="button primary full" href={result.checkpoint_url} download><ArrowDownToLine size={16}/>下載最佳權重</a>}
    {result?.training?.classes?.length > 0 && <div className="trained-class-info"><strong>微調類別 · {result.training.classes.length}</strong><p>{result.training.classes.slice(0, 16).join(' · ')}{result.training.classes.length > 16 ? ` · +${result.training.classes.length - 16}` : ''}</p><span>回到推論比較，在同模型的「模型權重」中選擇這次微調結果。</span></div>}
    {result?.artifact_urls && <div className="artifact-links">{Object.entries(result.artifact_urls).map(([name, url]) => <a key={name} href={url} download><FileText size={12}/>{name}</a>)}</div>}
  </div>;
}

export default function Training({ models, jobs, datasets, setDatasets, system, notify, submitJob, cancelJob, confirmLicense }) {
  const first = models.find(model => model.training_supported);
  const [modelId, setModelId] = useState(() => { try { return localStorage.getItem('vision-lab-train-model') || first?.id || ''; } catch { return first?.id || ''; } });
  const model = models.find(item => item.id === modelId) || first;
  const capability = model?.capabilities?.training || {};
  const [params, setParams] = useState({});
  const [format, setFormat] = useState('');
  const [path, setPath] = useState('');
  const [inputMethod, setInputMethod] = useState('path');
  const [datasetId, setDatasetId] = useState('');
  const [validation, setValidation] = useState(null);
  const [validating, setValidating] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  const [formatHelp, setFormatHelp] = useState(false);
  const [jobId, setJobId] = useState(null);
  const zipInput = useRef(null);
  const lastPositivePatience = useRef({});
  const configuredModel = useRef(null);
  const trainingJobs = jobs.filter(job => job.kind === 'training').sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)));
  const selectedJob = jobs.find(job => job.id === jobId) || trainingJobs[0];
  const dataset = datasets.find(dataset => dataset.id === datasetId);
  useEffect(() => {
    const current = models.find(item => item.id === modelId) || first; const defaults = current?.capabilities?.training?.defaults || {};
    if (configuredModel.current === current?.id) {
      setParams(previous => ({ ...defaults, ...Object.fromEntries(Object.entries(previous).filter(([key]) => Object.hasOwn(defaults, key))), ...(current?.license_accepted === true ? { accept_platform_license: true } : {}) }));
    } else {
      setParams({ ...defaults, ...(current?.license_accepted === true ? { accept_platform_license: true } : {}) }); setFormat(current?.capabilities?.training?.dataset_formats?.[0] || ''); setDatasetId(''); setValidation(null);
      configuredModel.current = current?.id;
    }
    try { if (current?.id) localStorage.setItem('vision-lab-train-model', current.id); } catch { /* Optional preference persistence. */ }
  }, [modelId, models]);
  const supported = capability.supported_params || Object.keys(capability.defaults || {});
  const licenseRequired = supported.includes('accept_platform_license');
  const licenseAccepted = model?.license_accepted === true;
  const licenseSpec = capability.parameters?.find(parameter => parameter.key === 'accept_platform_license');
  const earlyMonitor = capability.early_stopping_monitor || (model?.task === 'action' ? { metric: 'val_loss', mode: 'min', label: '驗證損失 val_loss' } : { metric: 'map50_95', mode: 'max', label: '驗證 mAP50–95' });
  const earlyMonitorLabel = earlyMonitor.label || earlyMonitor.metric || '模型的驗證指標';
  const earlyMonitorDirection = /越低越好|越高越好/.test(earlyMonitorLabel) ? '' : earlyMonitor.mode === 'min' ? '（越低越好）' : earlyMonitor.mode === 'max' ? '（越高越好）' : '';
  const earlyMonitorDescription = `監控：${earlyMonitorLabel}${earlyMonitorDirection}`;
  const patienceSpec = capability.parameters?.find(parameter => parameter.key === 'patience');
  const earlyStoppingEnabled = Number(params.patience) > 0;
  const patienceMaximum = patienceSpec?.max || 10000;
  const rememberedPatience = lastPositivePatience.current[model?.id] || Number(capability.defaults?.patience) || 5;
  const trainingNotConnected = capability.unavailable_reason || model?.training_reason || '此工作台尚未接入這個模型的微調流程，並不表示模型架構無法訓練。';
  function toggleEarlyStopping(enabled) {
    if (earlyStoppingEnabled) lastPositivePatience.current[model.id] = Number(params.patience);
    setParams(previous => ({ ...previous, patience: enabled ? Math.max(1, Math.min(patienceMaximum, rememberedPatience)) : 0 }));
  }
  function changePatience(event) {
    const value = Math.max(1, Math.min(patienceMaximum, Math.round(Number(event.target.value) || 1)));
    lastPositivePatience.current[model.id] = value;
    setParams(previous => ({ ...previous, patience: value }));
  }
  const choiceLists = { optimizer: capability.optimizers, loss: capability.losses, scheduler: capability.schedulers, strategy: capability.strategies };
  function fieldFor(key) {
    if (!supported.includes(key)) return null;
    const info = parameterInfo[key]; const values = choiceLists[key]; const definition = capability.parameters?.find(parameter => parameter.key === key); let param;
    if (definition) param = { ...definition, ...(definition.options ? { options: definition.options.map(value => { const option = optionParts(value); return { value: option.value, label: choiceNames[option.value] || option.label }; }) } : {}) };
    else if (values?.length) param = { key, label: stringLabels[key] || key, type: 'select', options: values.map(value => { const option = optionParts(value); return { value: option.value, label: choiceNames[option.value] || option.label }; }) };
    else if (typeof (capability.defaults || {})[key] === 'boolean') param = { key, label: stringLabels[key] || key, type: 'boolean' };
    else if (info || typeof (capability.defaults || {})[key] === 'number') param = { key, label: info?.[0] || key, type: 'number', min: info?.[1], max: info?.[2], step: info?.[3] || 'any' };
    else param = { key, label: stringLabels[key] || key, type: 'string' };
    return <ParameterField key={key} param={param} value={params[key]} onChange={value => setParams(previous => ({ ...previous, [key]: value }))}/>;
  }
  async function acceptValidation(data) {
    setValidation(data);
    if (data.valid && data.dataset) { setDatasets(previous => [data.dataset, ...previous.filter(item => item.id !== data.dataset.id)]); setDatasetId(data.dataset.id); notify('資料集驗證通過，可以開始設定訓練。'); }
    else { setDatasetId(''); notify('資料集未通過驗證，請修正下方問題。', true); }
  }
  async function validatePath() {
    if (!path.trim()) return notify('請輸入資料集的本機路徑。', true);
    setValidating(true); setValidation(null);
    try { await acceptValidation(await api('/datasets/validate', { method: 'POST', body: { path: path.trim(), format } })); }
    catch (error) { setValidation({ valid: false, errors: [error.message], warnings: [] }); setDatasetId(''); notify(error.message, true); }
    finally { setValidating(false); }
  }
  async function uploadZip(event) {
    const file = event.target.files?.[0]; event.target.value = ''; if (!file) return;
    setValidating(true); setValidation(null); const body = new FormData(); body.append('file', file); body.append('format', format);
    try { await acceptValidation(await api('/datasets/upload', { method: 'POST', body })); } catch (error) { setValidation({ valid: false, errors: [error.message], warnings: [] }); setDatasetId(''); notify(error.message, true); } finally { setValidating(false); }
  }
  async function startTraining() {
    if (!model?.training_supported || !dataset) return;
    if (!deviceAvailable(params.device, system)) { notify('選取的運算裝置目前不可用，請重新選擇。', true); return; }
    if (licenseRequired && !licenseAccepted && params.accept_platform_license !== true) { notify('請先閱讀模型授權條款，並勾選確認具備所需使用授權。', true); return; }
    const values = Object.fromEntries(supported.filter(key => params[key] !== undefined).map(key => [key, params[key]]));
    if (licenseRequired && licenseAccepted) values.accept_platform_license = true;
    setSubmitting(true);
    try { const job = await submitJob({ kind: 'training', model_id: model.id, dataset_id: dataset.id, params: values }); setJobId(job.id); }
    catch (error) { notify(error.message, true); } finally { setSubmitting(false); }
  }
  const basicKeys = ['epochs', 'batch_size', 'imgsz', 'lr', 'weight_decay', 'optimizer', 'loss', 'scheduler', 'strategy'];
  const extraKeys = supported.filter(key => !basicKeys.includes(key) && !['patience', 'accept_platform_license'].includes(key));
  const reason = !model?.available ? model?.reason || '模型環境尚未就緒。' : !model?.training_supported ? trainingNotConnected : !deviceAvailable(params.device, system) ? '選取的運算裝置目前不可用，請重新選擇。' : licenseRequired && !licenseAccepted && params.accept_platform_license !== true ? '請先確認 RF-DETR 2XL 的模型使用授權。' : !dataset ? '請先載入並驗證一份資料集。' : validating ? '正在驗證資料集。' : '';
  const formatInfo = {
    yolo_detection: ['YOLO 格式需要 dataset.yaml、訓練與驗證影像，以及對應的標註文字檔。', 'dataset.yaml\nimages/train/ · images/val/\nlabels/train/ · labels/val/\n\n# 每行標註：class_id x_center y_center width height\n# 座標以影像尺寸正規化為 0–1'],
    coco_detection: ['資料夾內需要 train.json 與 val.json，並包含 COCO bounding box 標註與影像。', 'dataset/\n  train.json\n  val.json\n  images/\n    image_001.jpg\n    image_002.jpg'],
    imagefolder: ['依類別分資料夾，train 與 val 必須包含一致的類別。', 'dataset/\n  train/\n    class_a/*.jpg\n    class_b/*.jpg\n  val/\n    class_a/*.jpg\n    class_b/*.jpg'],
    video_csv: ['需要 train.csv、val.csv 與 classes.txt。CSV 的 path 為相對資料集根目錄的影片路徑，label 為從 0 開始的類別編號。', '# train.csv / val.csv\npath,label\nvideos/walking_01.mp4,0\nvideos/running_01.mp4,1\n\n# classes.txt（每行一個類別名稱）\nwalking\nrunning']
  };
  return <main className="training-page scroll-page">
    <div className="page-intro"><div><span className="eyebrow"><span/>FINE-TUNING STUDIO</span><h1>把資料，變成你的模型<span className="accent-dot">。</span></h1><p>從監督資料集開始，控制訓練設定，觀察每一輪學習的變化。</p></div><span className="intro-badge"><Waypoints size={16}/>本機訓練 · 可重現設定</span></div>
    <div className="training-layout"><div className="training-configuration">
      <section className="config-section"><div className="section-heading"><span className="step-number">01</span><div><h2>選擇基礎模型</h2><p>模型決定輸入格式，以及可使用的訓練方法。</p></div><Settings2 size={19}/></div><div className="config-body"><label className="field"><span>預訓練模型</span><select value={model?.id || ''} onChange={event => setModelId(event.target.value)}>{models.map(item => <option key={item.id} value={item.id}>{item.name} · {taskNames[item.task]}{!item.training_supported ? '（微調尚未接入）' : ''}</option>)}</select></label><div className={`model-readiness ${model?.training_supported && model?.available ? '' : 'warning'}`}><span className="status-dot"/><span>{model?.training_supported ? model.available ? '微調介面已就緒' : model.reason || '模型環境尚未就緒。' : trainingNotConnected}</span>{model?.training_supported && model?.available && <span className="subtle">使用本機權重</span>}</div>
        {licenseRequired && !licenseAccepted && <section className={`training-license ${params.accept_platform_license ? 'accepted' : ''}`} aria-labelledby="training-license-heading"><div><CircleAlert size={17}/><h3 id="training-license-heading">模型使用授權確認</h3>{model?.license_url && <a href={model.license_url} target="_blank" rel="noreferrer">閱讀 PML 1.0 ↗</a>}</div><p>RF-DETR 2XL 使用 PML 1.0 授權。開始微調前，請確認你的使用方式符合授權條款。</p><label className="check-field"><input type="checkbox" checked={params.accept_platform_license === true} onChange={event => { const accepted = event.target.checked; setParams(previous => ({ ...previous, accept_platform_license: accepted })); if (accepted) confirmLicense(model.id); }}/><span>{licenseSpec?.label || '我已閱讀 PML 1.0，並確認具備所需使用授權。'}</span></label></section>}
        {licenseAccepted && model?.license_url && <a className="license-link" href={model.license_url} target="_blank" rel="noreferrer">授權已確認 · 閱讀條款 ↗</a>}
      </div></section>
      <section className="config-section"><div className="section-heading"><span className="step-number">02</span><div><h2>載入監督資料集</h2><p>先驗證結構、標籤與切分，再啟動訓練。</p></div><Database size={19}/></div><div className="config-body">
        <div className="dataset-top-row"><label className="field"><span>資料集格式</span><select disabled={!model?.training_supported} value={format} onChange={event => { setFormat(event.target.value); setDatasetId(''); setValidation(null); }}>{(capability.dataset_formats || []).map(value => <option key={value} value={value}>{formatNames[value] || value}</option>)}</select></label><button className="button ghost small" onClick={() => setFormatHelp(value => !value)}><FileText size={14}/>{formatHelp ? '收起格式範例' : '查看格式範例'}<ChevronDown size={13} className={formatHelp ? 'rotated' : ''}/></button></div>
        {formatHelp && <div className="format-help"><p>{formatInfo[format]?.[0] || '請選擇模型支援的資料集格式。'}</p><pre>{formatInfo[format]?.[1]}</pre></div>}
        <div className="dataset-input-method" role="group" aria-label="資料集載入方式"><button className={inputMethod === 'path' ? 'active' : ''} aria-pressed={inputMethod === 'path'} disabled={validating} onClick={() => setInputMethod('path')}><FolderOpen size={15}/>本機資料夾 / YAML</button><button className={inputMethod === 'zip' ? 'active' : ''} aria-pressed={inputMethod === 'zip'} disabled={validating} onClick={() => setInputMethod('zip')}><FileArchive size={15}/>上傳 ZIP</button></div>
        {inputMethod === 'path' ? <div className="dataset-input-panel"><p>直接讀取工作台所在電腦的資料，不需壓縮或上傳。請貼上資料夾或 YAML 的絕對路徑。</p><label className="field"><span>本機資料集路徑 <small>{format === 'yolo_detection' ? '指向 dataset.yaml' : '指向資料集根資料夾'}</small></span><div className="path-input"><FolderOpen size={17}/><input value={path} onChange={event => { setPath(event.target.value); setDatasetId(''); setValidation(null); }} placeholder={system?.platform?.system === 'Windows' ? format === 'yolo_detection' ? 'C:\\datasets\\my-dataset\\dataset.yaml' : 'C:\\datasets\\my-dataset' : format === 'yolo_detection' ? '/datasets/my-dataset/dataset.yaml' : '/datasets/my-dataset'} disabled={!model?.training_supported}/><button className="button secondary small" onClick={validatePath} disabled={validating || !format || !path.trim() || !model?.training_supported}>{validating ? <LoaderCircle className="spin" size={14}/> : <FileCheck2 size={14}/>}驗證</button></div></label></div> : <div className="dataset-input-panel"><p>將完整資料集打包為 ZIP，保留訓練／驗證切分、標註與類別資訊。上傳後會解壓並驗證格式。</p><div className="dataset-upload-row"><span>資料集 ZIP</span><button className="button secondary small" onClick={() => zipInput.current?.click()} disabled={validating || !format || !model?.training_supported}>{validating ? <LoaderCircle className="spin" size={14}/> : <Upload size={14}/>}選擇 ZIP 檔</button></div></div>}
        <input type="file" accept=".zip,application/zip" ref={zipInput} hidden onChange={uploadZip}/>
        {datasets.filter(item => item.format === format).length > 0 && <label className="field"><span>已驗證資料集</span><select value={datasetId} onChange={event => { setDatasetId(event.target.value); setValidation(null); }}><option value="">選擇一份已驗證資料集</option>{datasets.filter(item => item.format === format).map(item => <option key={item.id} value={item.id}>{item.name} · {item.train_count} train / {item.val_count} val</option>)}</select></label>}
        {validation?.errors?.length > 0 && <div className="validation-errors"><CircleAlert size={16}/><div><strong>需要修正的問題</strong>{validation.errors.map((error, index) => <p key={index}>{typeof error === 'string' ? error : JSON.stringify(error)}</p>)}</div></div>}
        {validation?.warnings?.length > 0 && <div className="validation-warnings">{validation.warnings.map((warning, index) => <p key={index}>{warning}</p>)}</div>}
        {dataset && <div className="dataset-success"><div><span><Check size={15}/>資料驗證通過</span><strong>{dataset.name}</strong></div><div className="dataset-stats"><span><b>{dataset.train_count}</b>訓練樣本</span><span><b>{dataset.val_count}</b>驗證樣本</span><span><b>{dataset.num_classes || dataset.classes?.length}</b>類別</span></div><p>{dataset.classes?.slice(0, 12).join(' · ')}{dataset.classes?.length > 12 ? ` · +${dataset.classes.length - 12}` : ''}</p></div>}
      </div></section>
      <section className="config-section"><div className="section-heading"><span className="step-number">03</span><div><h2>定義訓練方式</h2><p>只顯示這個模型實際支援的超參數與最佳化方法。</p></div><Gauge size={19}/></div><div className="config-body"><div className="training-params">{basicKeys.map(fieldFor)}</div>{!model?.training_supported && <p className="empty-text">選擇微調流程已接入的模型後，即可設定參數。</p>}
        {supported.includes('patience') && <section className={`early-stopping ${earlyStoppingEnabled ? 'enabled' : ''}`} aria-labelledby="early-stopping-heading"><div className="early-stopping-heading"><div><h3 id="early-stopping-heading">Early Stopping <span>提前停止</span></h3><p>連續幾輪沒有改善就停止；停用時跑到設定輪數。</p></div><label className="check-field"><input type="checkbox" checked={earlyStoppingEnabled} onChange={event => toggleEarlyStopping(event.target.checked)}/><span>啟用</span></label></div><div className="early-stopping-controls"><label className="field"><span>耐心輪數 <small>Patience</small></span><div className="patience-input"><input aria-label="Early Stopping 耐心輪數" type="number" min={1} max={patienceMaximum} step={1} disabled={!earlyStoppingEnabled} value={earlyStoppingEnabled ? params.patience : rememberedPatience} onChange={changePatience}/><span>輪</span></div></label><p>{earlyMonitorDescription}<span>{earlyStoppingEnabled ? `連續 ${params.patience} 輪未改善時，自動結束訓練。` : '目前已停用（patience = 0）。'}</span></p></div></section>}
        {extraKeys.length > 0 && <><button className="advanced-toggle" onClick={() => setAdvanced(value => !value)}><SlidersIcon/>進階設定 <span>{extraKeys.length} 個參數</span><ChevronDown size={16} className={advanced ? 'rotated' : ''}/></button>{advanced && <div className="training-params advanced-fields">{extraKeys.map(fieldFor)}</div>}</>}
        <div className="training-note"><CircleAlert size={14}/><p>損失函數與最佳化器會依模型架構開放。訓練工作依序執行，權重與設定會保存在本機。</p></div><div className="training-submit"><span>{reason || `${dataset?.name} · ${params.epochs || '—'} epochs · ${params.optimizer || '預設最佳化器'}`}</span><button className="button primary" disabled={Boolean(reason) || submitting} onClick={startTraining}>{submitting ? <LoaderCircle size={15} className="spin"/> : <Play size={15} fill="currentColor"/>}開始訓練</button></div>
      </div></section>
    </div><aside className="training-right"><JobMonitor job={selectedJob} cancelJob={cancelJob} models={models}/><section className="recent-training"><div className="panel-heading"><span>訓練實驗</span><b>{trainingJobs.length}</b></div>{!trainingJobs.length ? <p className="empty-text">尚未建立訓練實驗。<br/>你的每一次嘗試都會保留在這裡。</p> : trainingJobs.slice(0, 15).map(job => <button className={`training-history-item ${selectedJob?.id === job.id ? 'selected' : ''}`} key={job.id} onClick={() => setJobId(job.id)}><div><strong>{models.find(model => model.id === job.model_id)?.name || job.model_id}</strong><small>{new Date(job.created_at).toLocaleString('zh-TW', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })} · {job.params?.epochs || '—'} epochs</small></div><span className={`job-status ${job.status}`}><i/>{statusNames[job.status]}</span></button>)}</section></aside></div>
  </main>;
}
function SlidersIcon() { return <Settings2 size={15}/>; }
