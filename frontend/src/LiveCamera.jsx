import React, { useEffect, useRef, useState } from 'react';
import { Camera, CameraOff, ChevronDown, CircleHelp, ExternalLink, LoaderCircle, Monitor, Play, Plus, RefreshCw, Square, Trash2, Wifi } from 'lucide-react';
import { api, modelDefaults, statusNames, taskNames, uid } from './api';
import { ParameterField } from './InferenceCanvas';
import { parseVdoViewUrl, VdoFrameSource } from './vdoSource';
import { advanceFrameRate, livePollInterval } from './liveMetrics';
import { parseLiveDuration, formatLiveDuration } from './liveDuration';
import { VdoPeerSource } from './vdoPeerSource';
import { recordPipelineSample, summarizePipeline } from './liveDiagnostics';
import { LatestFramePipeline } from './livePipeline';
import { LiveJpegEncoder } from './liveJpegEncoder';
import { canRunWithWeights, checkpointChoices, checkpointRequest, checkpointUnavailable, selectedCheckpoint } from './checkpoints';
import { deviceAvailable } from './systemDevices';

const inactiveStatuses = new Set(['completed', 'failed', 'cancelled']);
const fileOnlyParams = new Set(['frame_stride', 'max_frames', 'num_clips', 'start_seconds']);
const measuredMs = value => Number.isFinite(value) ? `${value.toFixed(value < 10 ? 1 : 0)} ms` : '—';
const stopRemote = id => id && fetch(`/api/live/sessions/${encodeURIComponent(id)}/stop`, { method: 'POST', keepalive: true }).catch(() => {});
const cameraError = error => ({
  NotAllowedError: '攝影機權限未開啟。請允許此網站使用攝影機；若目前瀏覽器無法授權，可在 Chrome／Edge 開啟 http://127.0.0.1:8765/。',
  NotFoundError: '找不到攝影機。請接上 USB 攝影機，或先啟動 iVCam／OBS 的虛擬攝影機，再重新整理裝置。',
  NotReadableError: '無法讀取攝影機。裝置可能被其他程式占用；請確認 iVCam／OBS 已啟動，並關閉其他正在使用攝影機的程式。',
  OverconstrainedError: '所選攝影機目前不可用。請重新整理裝置，選擇另一個攝影機後再開啟。',
  AbortError: '攝影機開啟中斷，請重新開啟。',
}[error?.name] || error?.message || '攝影機無法開啟，請檢查裝置連線與瀏覽器權限。');

function ModelCard({ card, index, models, jobs, registeredCheckpoints, confirmLicense, disabled, result, finished, onChange, onRemove }) {
  const [advanced, setAdvanced] = useState(false);
  const model = models.find(item => item.id === card.modelId);
  const checkpoints = checkpointChoices(card.modelId, registeredCheckpoints, jobs);
  const chosenCheckpoint = checkpoints.find(item => item.id === selectedCheckpoint(card));
  const training = chosenCheckpoint?.training;
  const trainedPrompt = training?.caption || (training?.classes?.length ? `${training.classes.join(' . ')} .` : '');
  const allParams = (model?.capabilities?.inference_params || []).filter(param => !fileOnlyParams.has(param.key) && (param.key !== 'accept_platform_license' || model?.license_accepted !== true));
  const params = advanced ? allParams : allParams.filter(param => ['conf', 'prompt', 'clip_seconds', 'top_k', 'accept_platform_license'].includes(param.key));
  const busy = result?.status === 'loading' || result?.status === 'warming';
  const label = finished ? '本次任務最後結果' : result?.stale ? '等待新結果' : ({ loading: '載入模型', warming: '累積影片片段', running: '即時辨識', completed: '本次任務最後結果', failed: '模型失敗' }[result?.status] || '等待開始');
  return <article className={`live-model-card ${result?.status === 'failed' ? 'has-error' : ''}`}>
    <header><span className="node-number">{String(index + 1).padStart(2, '0')}</span><div><strong>{model?.name || '選擇模型'}</strong><small>{taskNames[model?.task] || '即時比較區塊'}</small></div><button className="icon-button danger-hover" aria-label={`移除即時區塊 ${index + 1}`} disabled={disabled} onClick={onRemove}><Trash2 size={15}/></button></header>
    <div className="live-card-config">
      <label className="field"><span>辨識模型</span><select value={card.modelId} disabled={disabled} onChange={event => { const next = models.find(item => item.id === event.target.value); onChange({ modelId: next.id, params: modelDefaults(next), checkpointJobId: '', checkpointId: '' }); }}>{models.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>
      {(checkpoints.length > 0 || selectedCheckpoint(card)) && <label className="field"><span>模型權重</span><select value={selectedCheckpoint(card)} disabled={disabled} onChange={event => onChange({ checkpointId: event.target.value, checkpointJobId: '' })}><option value="" disabled={!model?.available}>{model?.available ? '原始預訓練權重' : '原始權重未齊，請選擇微調權重'}</option>{selectedCheckpoint(card) && !chosenCheckpoint && <option value={selectedCheckpoint(card)} disabled>先前權重已不可用</option>}{checkpoints.map(item => <option key={item.id} value={item.id} disabled={!item.valid}>{item.source === 'external' ? '外部 · ' : ''}{item.name}{!item.valid ? ' · 路徑失效' : ''}</option>)}</select></label>}
      {card.modelId === 'grounding_dino_base' && trainedPrompt && <button className="button ghost small" disabled={disabled || trainedPrompt.length > 2000} onClick={() => onChange({ params: { ...card.params, prompt: trainedPrompt } })}>套用微調類別為偵測目標</button>}
      <div className="node-params">{params.map(param => <ParameterField key={param.key} compact disabled={disabled} param={param} value={card.params[param.key]} onChange={value => { onChange({ params: { ...card.params, [param.key]: value } }); if (param.key === 'accept_platform_license' && value === true) confirmLicense(model.id); }}/>)}</div>
      {model?.license_url && <a className="license-link" href={model.license_url} target="_blank" rel="noreferrer">{model.license_accepted === true ? '授權已確認 · 閱讀條款 ↗' : '閱讀模型授權條款 ↗'}</a>}
      {allParams.length > params.length || advanced ? <button className="parameter-toggle" onClick={() => setAdvanced(value => !value)}>{advanced ? '收起進階參數' : '更多模型參數'}<ChevronDown size={13} className={advanced ? 'rotated' : ''}/></button> : null}
    </div>
    <div className="live-result-preview">{result?.image_url ? <img src={result.image_url} alt={`${model?.name} ${finished ? '最後' : '即時'}辨識結果`}/> : <div className="empty-preview">{busy ? <LoaderCircle size={28} className="spin"/> : <Monitor size={30} strokeWidth={1.2}/>}<strong>{label}</strong><span>{result?.message || (model?.task === 'action' ? `開始後先累積 ${card.params.clip_seconds || 4} 秒影像，再持續辨識行為` : '開啟攝影機，按下開始即時辨識')}</span></div>}<span className={`preview-label ${!finished && result?.status === 'running' ? 'result' : ''}`}>{label}</span></div>
    <div className="live-card-result">
      {result?.status === 'failed' ? <p className="inline-error">{result.message || result.error || '模型執行失敗，請停止後調整設定。'}</p> : <>
        <div className="live-result-metrics"><span>單次推論 <b>{Number.isFinite(result?.inference_ms) ? `${result.inference_ms.toFixed(0)} ms` : '—'}</b></span><span title="最近約 4 秒內，模型實際完成辨識的影格數／秒">辨識更新 <b>{Number.isFinite(result?.update_hz) ? `${result.update_hz.toFixed(1)} FPS` : '—'}</b></span>{result?.sequence != null && <span>影格 <b>#{result.sequence}</b></span>}</div>
        {model?.task === 'action' && result && <p className="live-window-info">{result.status === 'warming' ? result.message : `滾動片段 ${result.window_seconds ?? card.params.clip_seconds} 秒${result.sampled_frames != null ? ` · ${result.sampled_frames} 個取樣影格${Number.isFinite(result.unique_frames) && result.unique_frames < result.sampled_frames ? `（實際 ${result.unique_frames} 張）` : ''}` : ''}`}</p>}
        {!finished && result?.stale && <p className="live-window-info">結果已 {result.age_seconds} 秒未更新；目前顯示上一個完成的影格。</p>}
        <div className="prediction-list">{(result?.predictions || []).slice(0, 12).map((prediction, i) => <span key={i}>{prediction.label}<b>{Number.isFinite(prediction.score) ? `${(prediction.score * 100).toFixed(1)}%` : ''}</b></span>)}</div>
        {result?.status === 'running' && !result.predictions?.length && <p className="result-placeholder">這個影格沒有超過閾值的辨識結果。</p>}
      </>}
    </div>
  </article>;
}

export default function LiveCamera({ models, jobs, system, confirmLicense, checkpoints = [] }) {
  const availableModels = models.filter(model => model.inference_supported && ['detection', 'action'].includes(model.task) && (model.available || model.environment_available && checkpointChoices(model.id, checkpoints, jobs).some(item => item.valid)));
  const [cards, setCards] = useState(() => { const model = availableModels.find(item => item.id === 'yolo26x') || availableModels[0]; return model ? [{ id: uid(), modelId: model.id, params: modelDefaults(model), checkpointJobId: '' }] : []; });
  const [devices, setDevices] = useState([]), [deviceId, setDeviceId] = useState('');
  const [sourceMode, setSourceMode] = useState('camera'), [vdoUrl, setVdoUrl] = useState(''), [vdoSource, setVdoSource] = useState(null);
  const [vdoMode, setVdoMode] = useState('direct'), [receiverStats, setReceiverStats] = useState(null);
  const [cameraStatus, setCameraStatus] = useState('idle'), [error, setError] = useState('');
  const [captureFps, setCaptureFps] = useState(30), [mirror, setMirror] = useState(false), [session, setSession] = useState(null), [liveState, setLiveState] = useState(null);
  const [durationMode, setDurationMode] = useState('custom'), [durationValue, setDurationValue] = useState('15'), [durationUnit, setDurationUnit] = useState('minutes');
  const [starting, setStarting] = useState(false), [stopping, setStopping] = useState(false), [captureStats, setCaptureStats] = useState({ count: 0, fps: 0 }), [connectionHelp, setConnectionHelp] = useState(false);
  const [diagnostics, setDiagnostics] = useState(() => summarizePipeline(null, 0));
  const [pageVisibility, setPageVisibility] = useState(() => globalThis.document?.visibilityState || 'unknown');
  const videoRef = useRef(null), canvasRef = useRef(null), streamRef = useRef(null), sessionRef = useRef(null);
  const iframeRef = useRef(null), vdoRef = useRef(null);
  const vdoPeerRef = useRef(null), receiverStatsTimerRef = useRef(null), receiverStatsSourceRef = useRef(null), videoReadyCleanupRef = useRef(null);
  const pipelineRef = useRef(null), diagnosticsUpdatedRef = useRef(0), framePipelineRef = useRef(null), jpegEncoderRef = useRef(null);
  const mountedRef = useRef(true), generationRef = useRef(0), pollTimerRef = useRef(null);
  const startingRef = useRef(false), captureRateRef = useRef(null), sentCountRef = useRef(0), ratesRef = useRef(new Map());
  const active = starting || stopping || Boolean(session && !inactiveStatuses.has(session.status));
  const duration = parseLiveDuration(durationValue, durationUnit, durationMode);

  function clearLoops() { framePipelineRef.current?.stop(); framePipelineRef.current = null; jpegEncoderRef.current?.dispose(); jpegEncoderRef.current = null; clearTimeout(pollTimerRef.current); }
  function resetDiagnostics() { pipelineRef.current = null; diagnosticsUpdatedRef.current = 0; setDiagnostics(summarizePipeline(null, 0)); }
  function releaseCamera() {
    clearTimeout(receiverStatsTimerRef.current); receiverStatsSourceRef.current = null; videoReadyCleanupRef.current?.(); videoReadyCleanupRef.current = null;
    const stream = streamRef.current; streamRef.current = null; stream?.getTracks().forEach(track => { track.onended = null; track.stop(); }); if (videoRef.current) videoRef.current.srcObject = null;
    const remote = vdoRef.current; vdoRef.current = null; remote?.stop();
    const peer = vdoPeerRef.current; vdoPeerRef.current = null; if (peer) Promise.resolve(peer.stop()).catch(() => {});
    if (mountedRef.current) { setVdoSource(null); setReceiverStats(null); }
  }
  async function stopAll(message = '') {
    const generation = ++generationRef.current; clearLoops(); releaseCamera(); const id = sessionRef.current;
    if (mountedRef.current) { setCameraStatus('idle'); setCaptureStats(previous => ({ ...previous, fps: 0 })); setStarting(false); setStopping(Boolean(id)); if (message) setError(message); }
    if (id) {
      try {
        const response = await api(`/live/sessions/${encodeURIComponent(id)}/stop`, { method: 'POST' }), value = response.session || response.job || response;
        if (!mountedRef.current || generation !== generationRef.current) return;
        setSession(value);
        if (!inactiveStatuses.has(value.status)) { schedulePoll(id, generation); return; }
      } catch (failure) { if (mountedRef.current && generation === generationRef.current) { setError(`已關閉攝影機；停止任務時連線失敗：${failure.message}。伺服器會在收不到影格後自動結束。`); setSession(prev => prev ? { ...prev, status: 'cancelled' } : null); } }
    }
    if (generation === generationRef.current) { sessionRef.current = null; if (mountedRef.current) setStopping(false); }
  }
  useEffect(() => {
    mountedRef.current = true;
    function leave() { generationRef.current += 1; clearLoops(); releaseCamera(); stopRemote(sessionRef.current); sessionRef.current = null; }
    function visibilityChanged() { setPageVisibility(document.visibilityState); }
    window.addEventListener('pagehide', leave);
    document.addEventListener('visibilitychange', visibilityChanged);
    return () => { mountedRef.current = false; window.removeEventListener('pagehide', leave); document.removeEventListener('visibilitychange', visibilityChanged); leave(); };
  }, []);
  useEffect(() => {
    if (!vdoSource) return;
    const generation = vdoSource.generation;
    if (vdoSource.mode === 'direct') {
      let remote, attachedStream, attachment;
      const current = () => mountedRef.current && generation === generationRef.current && vdoPeerRef.current === remote;
      const fail = failure => {
        if (!current()) return;
        const message = `VDO.Ninja 直接接收中斷：${failure?.message || '請確認手機仍在分享，或改用相容模式。'}`;
        if (sessionRef.current) { stopAll(message); return; }
        releaseCamera(); setCameraStatus('error'); setError(message);
      };
      const connectionTimeout = setTimeout(() => fail(new Error('30 秒內沒有收到可播放的影像。請確認手機仍在分享，或改用相容模式。')), 30000);
      const attachStream = stream => {
        if (!current()) return Promise.resolve();
        if (stream === attachedStream) return attachment;
        attachedStream = stream;
        attachment = (async () => {
          const video = videoRef.current;
          if (!video) throw new Error('找不到遠端影像預覽元件。');
          videoReadyCleanupRef.current?.(); video.srcObject = stream;
          let resolveReady, rejectReady, settled = false;
          const ready = new Promise((resolve, reject) => { resolveReady = resolve; rejectReady = reject; });
          const cleanup = () => { video.removeEventListener('loadeddata', checkReady); video.removeEventListener('canplay', checkReady); video.removeEventListener('resize', checkReady); video.removeEventListener('error', onVideoError); if (videoReadyCleanupRef.current === cancelReady) videoReadyCleanupRef.current = null; };
          const finish = error => { if (settled) return; settled = true; cleanup(); if (error) rejectReady(error); else resolveReady(); };
          const checkReady = () => { if (current() && video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0) finish(); };
          const onVideoError = () => finish(new Error('瀏覽器無法播放收到的遠端影像。'));
          const cancelReady = () => finish(new DOMException('遠端影像已停止。', 'AbortError'));
          videoReadyCleanupRef.current = cancelReady;
          video.addEventListener('loadeddata', checkReady); video.addEventListener('canplay', checkReady); video.addEventListener('resize', checkReady); video.addEventListener('error', onVideoError);
          try { await Promise.all([Promise.resolve().then(() => video.play()).then(checkReady), ready]); }
          finally { cleanup(); }
          if (!current()) return;
          clearTimeout(connectionTimeout); setCameraStatus('ready'); scheduleReceiverStats(remote, generation);
        })();
        return attachment;
      };
      try {
        remote = new VdoPeerSource({ source: vdoSource.parsed, onStream: stream => { Promise.resolve(attachStream(stream)).catch(fail); }, onError: fail });
        vdoPeerRef.current = remote;
        Promise.resolve(remote.start()).then(attachStream).catch(fail);
      } catch (failure) {
        clearTimeout(connectionTimeout);
        if (mountedRef.current && generation === generationRef.current) { releaseCamera(); setCameraStatus('error'); setError(`無法啟動 VDO.Ninja 直接接收：${failure?.message || '接收設定無效。'} 可改選相容模式後重新連接。`); }
        return;
      }
      return () => { clearTimeout(connectionTimeout); clearTimeout(receiverStatsTimerRef.current); videoReadyCleanupRef.current?.(); if (vdoPeerRef.current === remote) vdoPeerRef.current = null; Promise.resolve(remote.stop()).catch(() => {}); };
    }
    if (!iframeRef.current) return;
    const remote = new VdoFrameSource({ iframe: iframeRef.current, source: vdoSource.parsed });
    vdoRef.current = remote; remote.start();
    remote.getFrame(30000).then(() => {
      if (mountedRef.current && generation === generationRef.current && vdoRef.current === remote) setCameraStatus('ready');
    }).catch(failure => {
      if (!mountedRef.current || generation !== generationRef.current || vdoRef.current !== remote) return;
      releaseCamera(); setCameraStatus('error'); setError(`VDO.Ninja 連線失敗：${failure.message || '30 秒內沒有收到影像。請確認手機仍在分享，並使用 VIEW 觀看連結。'}`);
    });
    return () => { if (vdoRef.current === remote) vdoRef.current = null; remote.stop(); };
  }, [vdoSource]);

  function scheduleReceiverStats(remote, generation) {
    if (receiverStatsSourceRef.current === remote) return;
    receiverStatsSourceRef.current = remote;
    clearTimeout(receiverStatsTimerRef.current);
    async function pollStats() {
      if (!mountedRef.current || generation !== generationRef.current || vdoPeerRef.current !== remote) return;
      let stats;
      try { stats = await remote.getStats(); } catch { stats = null; }
      if (!mountedRef.current || generation !== generationRef.current || vdoPeerRef.current !== remote) return;
      setReceiverStats(stats);
      receiverStatsTimerRef.current = setTimeout(pollStats, 1000);
    }
    pollStats();
  }

  function changeSourceMode(mode) {
    if (active || cameraStatus === 'opening' || sourceMode === mode) return;
    generationRef.current += 1; releaseCamera(); resetDiagnostics(); setSourceMode(mode); setCameraStatus('idle'); setError(''); setCaptureStats(previous => ({ ...previous, fps: 0 }));
  }
  function connectVdo() {
    if (active || cameraStatus === 'opening') return;
    let parsed;
    try { parsed = parseVdoViewUrl(vdoUrl); } catch (failure) { setError(failure.message); return; }
    const generation = ++generationRef.current; releaseCamera(); resetDiagnostics(); setError(''); setCameraStatus('opening'); setVdoSource({ parsed, generation, mode: vdoMode });
  }
  function changeVdoMode(mode) {
    if (active || cameraStatus === 'opening' || mode === vdoMode) return;
    generationRef.current += 1; releaseCamera(); resetDiagnostics(); setVdoMode(mode); setCameraStatus('idle'); setError('');
  }

  async function listDevices() {
    if (!navigator.mediaDevices?.enumerateDevices) return setError('目前瀏覽器沒有提供攝影機介面。請在此電腦的 Chrome／Edge 開啟 http://127.0.0.1:8765/。');
    try { const list = (await navigator.mediaDevices.enumerateDevices()).filter(item => item.kind === 'videoinput'); if (mountedRef.current) setDevices(list); }
    catch (failure) { if (mountedRef.current) setError(cameraError(failure)); }
  }
  async function openCamera() {
    if (active || cameraStatus === 'opening') return;
    if (!navigator.mediaDevices?.getUserMedia) return setError('目前瀏覽器沒有提供攝影機介面。請在此電腦的 Chrome／Edge 開啟 http://127.0.0.1:8765/，再允許攝影機權限。');
    const generation = ++generationRef.current; releaseCamera(); resetDiagnostics(); setCameraStatus('opening'); setError('');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { ...(deviceId ? { deviceId: { exact: deviceId } } : {}), width: { ideal: 960 }, height: { ideal: 540 }, frameRate: { ideal: 30, max: 30 } }, audio: false });
      if (!mountedRef.current || generation !== generationRef.current) { stream.getTracks().forEach(track => track.stop()); return; }
      streamRef.current = stream; const video = videoRef.current; video.srcObject = stream; await video.play();
      if (!mountedRef.current || generation !== generationRef.current) { stream.getTracks().forEach(track => track.stop()); return; }
      stream.getVideoTracks().forEach(track => { track.onended = () => stopAll('攝影機連線已中斷。重新接上裝置後，請再次開啟攝影機。'); });
      const actualDevice = stream.getVideoTracks()[0]?.getSettings().deviceId; if (actualDevice) setDeviceId(actualDevice);
      setCameraStatus('ready'); await listDevices();
    } catch (failure) { if (mountedRef.current && generation === generationRef.current) { releaseCamera(); setCameraStatus('error'); setError(cameraError(failure)); } }
  }

  function scheduleCapture(id, generation, fps) {
    const current = () => mountedRef.current && generation === generationRef.current && sessionRef.current === id;
    const ensureCurrent = () => { if (!current()) throw new DOMException('影格傳送已停止。', 'AbortError'); };
    framePipelineRef.current?.stop();
    jpegEncoderRef.current?.dispose();
    const encoder = new LiveJpegEncoder({ getFallbackCanvas: () => canvasRef.current });
    jpegEncoderRef.current = encoder;
    const pipeline = new LatestFramePipeline({
      fps,
      async capture({ scheduledAt, startedAt }) {
        ensureCurrent();
        const video = videoRef.current, timings = { timer_lag_ms: Math.max(0, startedAt - scheduledAt) };
        let blob;
        if (vdoRef.current) { const remote = vdoRef.current; blob = await remote.getFrame(); Object.assign(timings, remote.lastTimings || {}); }
        else {
          const convertStarted = performance.now();
          blob = await encoder.encode(video);
          timings.convert_ms = performance.now() - convertStarted;
        }
        ensureCurrent();
        if (!blob || blob.size > 1024 * 1024) throw new Error('攝影機影格無法轉換成 1 MB 以下的 JPEG，請改用較低解析度的來源。');
        return { blob, timings };
      },
      async upload({ blob }, signal) {
        ensureCurrent();
        const form = new FormData(); form.append('frame', blob, 'frame.jpg');
        await api(`/live/sessions/${encodeURIComponent(id)}/frames`, { method: 'POST', body: form, signal });
      },
      onAccepted({ blob, timings }, { captureStartedAt, uploadStartedAt, completedAt }) {
        if (!current()) return;
        pipelineRef.current = recordPipelineSample(pipelineRef.current, { ...timings, upload_ms: completedAt - uploadStartedAt, jpeg_bytes: blob.size, cycle_ms: completedAt - captureStartedAt }, completedAt);
        sentCountRef.current += 1;
      },
      onError(failure) {
        if (current()) void stopAll(['AbortError', 'TimeoutError'].includes(failure.name) ? '影格傳送逾時，已停止攝影機。請確認本機服務後重新開始。' : `即時影像已停止：${failure.message}`);
      },
    });
    framePipelineRef.current = pipeline;
    pipeline.start();
  }
  function schedulePoll(id, generation) {
    let failures = 0;
    const interval = livePollInterval(captureFps);
    async function poll() {
      if (!mountedRef.current || generation !== generationRef.current || sessionRef.current !== id) return;
      const started = performance.now(); let nextInterval = interval;
      try {
        const response = await api(`/live/sessions/${encodeURIComponent(id)}`), value = response.session || response.job || response;
        if (!mountedRef.current || generation !== generationRef.current || sessionRef.current !== id) return;
        failures = 0; setSession(value);
        const now = performance.now(), sourceConnected = Boolean(streamRef.current || vdoRef.current || vdoPeerRef.current?.stream);
        if (sourceConnected && now - diagnosticsUpdatedRef.current >= 500) { setDiagnostics(summarizePipeline(pipelineRef.current, now)); diagnosticsUpdatedRef.current = now; }
        captureRateRef.current = advanceFrameRate(captureRateRef.current, sentCountRef.current, now);
        setCaptureStats({ count: sentCountRef.current, fps: sourceConnected ? captureRateRef.current?.rate ?? 0 : 0 });
        if (value.live_state?.status === 'loading' || !sourceConnected) nextInterval = 500;
        if (value.live_state) {
          const results = (value.live_state.models || []).map(result => {
            const measured = advanceFrameRate(ratesRef.current.get(result.key), Number(result.processed_frames), now);
            ratesRef.current.set(result.key, measured);
            return { ...result, update_hz: measured?.rate, age_seconds: measured?.ageSeconds ?? 0, stale: result.status === 'running' && measured?.ageSeconds >= 5 };
          });
          setLiveState({ ...value.live_state, models: results });
        }
        if (inactiveStatuses.has(value.status)) { generationRef.current += 1; sessionRef.current = null; clearLoops(); releaseCamera(); setCameraStatus('idle'); setCaptureStats(previous => ({ ...previous, fps: 0 })); setStopping(false); setStarting(false); if (value.status === 'failed') setError(value.error || value.message || '即時任務失敗，請調整模型或參數後再試。'); return; }
      } catch (failure) { failures += 1; if (failures >= 3) { if (mountedRef.current && generation === generationRef.current) await stopAll(`無法取得即時任務狀態：${failure.message}`); return; } }
      if (mountedRef.current && generation === generationRef.current && sessionRef.current === id) pollTimerRef.current = setTimeout(poll, failures ? 500 : Math.max(0, nextInterval - (performance.now() - started)));
    }
    poll();
  }
  async function startLive() {
    if (active || startingRef.current || cameraStatus !== 'ready' || !cards.length || duration.error || checkpointsMissing || unavailableDevice || weightsMissing) return;
    startingRef.current = true; setStarting(true); setError(''); setLiveState(null); setSession(null);
    const generation = generationRef.current;
    try {
      const response = await api('/live/sessions', { method: 'POST', body: { models: cards.map(card => ({ model_id: card.modelId, params: { ...card.params, ...(availableModels.find(model => model.id === card.modelId)?.license_accepted === true ? { accept_platform_license: true } : {}) }, ...checkpointRequest(card) })), capture_fps: captureFps, max_duration_seconds: duration.seconds } });
      const value = response.session || response.job || response;
      if (!value.id) throw new Error('伺服器沒有回傳即時任務 ID。');
      if (!mountedRef.current || generation !== generationRef.current || (!streamRef.current && !vdoRef.current && !vdoPeerRef.current?.stream)) { stopRemote(value.id); return; }
      sessionRef.current = value.id; setSession(value); sentCountRef.current = 0; captureRateRef.current = advanceFrameRate(null, 0, performance.now()); ratesRef.current.clear(); setCaptureStats({ count: 0, fps: 0 }); resetDiagnostics();
      scheduleCapture(value.id, generation, captureFps); schedulePoll(value.id, generation);
    } catch (failure) { if (mountedRef.current && generation === generationRef.current) setError(failure.message); }
    finally { startingRef.current = false; if (mountedRef.current && generation === generationRef.current) setStarting(false); }
  }
  function updateCard(id, patch) { setCards(previous => previous.map(card => card.id === id ? { ...card, ...patch } : card)); setLiveState(null); setSession(null); }
  function addCard() { const model = availableModels.find(item => !cards.some(card => card.modelId === item.id)) || availableModels[0]; if (model && cards.length < 4) { setCards(previous => [...previous, { id: uid(), modelId: model.id, params: modelDefaults(model), checkpointJobId: '' }]); setLiveState(null); setSession(null); } }
  const licenseMissing = cards.some(card => { const model = availableModels.find(model => model.id === card.modelId); return model?.license_url && model.license_accepted !== true && !card.params.accept_platform_license; });
  const checkpointsMissing = cards.some(card => checkpointUnavailable(card, checkpointChoices(card.modelId, checkpoints, jobs)));
  const unavailableDevice = cards.some(card => !deviceAvailable(card.params.device, system));
  const weightsMissing = cards.some(card => !canRunWithWeights(availableModels.find(model => model.id === card.modelId), card, checkpointChoices(card.modelId, checkpoints, jobs)));
  const finished = Boolean(session && inactiveStatuses.has(session.status));
  const sessionLabel = starting ? '建立即時任務' : stopping ? '正在停止' : session && !finished ? (liveState?.status === 'loading' ? '載入模型中' : '即時任務執行中') : cameraStatus === 'ready' ? (sourceMode === 'vdo' ? 'VDO.Ninja 已連線' : '攝影機預覽中') : cameraStatus === 'opening' && sourceMode === 'vdo' ? '等待 VDO.Ninja 影像' : '影像來源未開啟';
  return <main className="scroll-page live-page">
    <div className="page-intro"><div><span className="eyebrow"><span/>LIVE VISION</span><h1>接上鏡頭，觀察當下<span className="accent-dot">。</span></h1><p>USB、手機虛擬攝影機與遠端影像，共用一個即時比較畫面。</p></div><button className="button secondary" onClick={() => setConnectionHelp(value => !value)}><CircleHelp size={16}/>{connectionHelp ? '收起接入說明' : '手機與跨網路接入'}</button></div>
    {connectionHelp && <section className="live-connection-help"><div><Camera size={20}/><h3>USB／同一個 Wi-Fi</h3><p>手機安裝 iVCam，電腦安裝 Windows 用戶端。連線成功後，開啟攝影機並選取 iVCam。</p><a href="https://www.e2esoft.com/ivcam/" target="_blank" rel="noreferrer">iVCam 官方下載與說明 <ExternalLink size={12}/></a></div><div><Wifi size={20}/><h3>VDO.Ninja 直接接入</h3><p>手機以 VDO.Ninja 分享鏡頭，將 VIEW 觀看連結貼入左側「VDO.Ninja 連結」。按下連接，看到影像後即可開始辨識。手機需保持分享頁面開啟。</p><a href="https://docs.vdo.ninja/getting-started/mobile-phone-camera-into-webcam" target="_blank" rel="noreferrer">VDO.Ninja 官方操作說明 <ExternalLink size={12}/></a></div><div><Monitor size={20}/><h3>跨網路影像，本機運算</h3><p>手機可以使用另一個網路或 4G、5G，模型仍在這台電腦執行。若已用 OBS 轉接，選「攝影機裝置」中的 OBS Virtual Camera 即可。</p><span>跨網路延遲與穩定度取決於連線品質。</span></div></section>}
    <div className="live-layout">
      <aside className="live-source-panel">
        <div className="panel-heading"><span><Camera size={17}/>影像來源</span><span className={`live-source-status ${cameraStatus === 'ready' ? 'ready' : ''}`}><i/>{cameraStatus === 'ready' ? '已連線' : cameraStatus === 'opening' ? '連接中' : '未連線'}</span></div>
        <div className="live-source-mode" role="group" aria-label="影像來源類型"><button className={sourceMode === 'camera' ? 'selected' : ''} aria-pressed={sourceMode === 'camera'} disabled={active || cameraStatus === 'opening'} onClick={() => changeSourceMode('camera')}><Camera size={14}/>攝影機裝置</button><button className={sourceMode === 'vdo' ? 'selected' : ''} aria-pressed={sourceMode === 'vdo'} disabled={active || cameraStatus === 'opening'} onClick={() => changeSourceMode('vdo')}><Wifi size={14}/>VDO.Ninja 連結</button></div>
        <div className={`live-source-video ${sourceMode === 'vdo' ? 'is-vdo' : ''}`}>
          {sourceMode === 'camera' || vdoMode === 'direct' ? <video ref={videoRef} autoPlay muted playsInline className={sourceMode === 'camera' && mirror ? 'mirrored' : ''}/> : vdoSource && <iframe key={vdoSource.generation} ref={iframeRef} src={vdoSource.parsed.url} title="VDO.Ninja 遠端影像預覽" allow="autoplay; fullscreen" referrerPolicy="no-referrer"/>}
          {cameraStatus !== 'ready' && !(sourceMode === 'vdo' && vdoSource) && <div className="empty-preview">{cameraStatus === 'opening' ? <LoaderCircle className="spin" size={30}/> : sourceMode === 'vdo' ? <Wifi size={30} strokeWidth={1.2}/> : <CameraOff size={30} strokeWidth={1.2}/>}<strong>{sourceMode === 'vdo' ? '等待遠端影像' : cameraStatus === 'opening' ? '等待攝影機權限' : '攝影機目前關閉'}</strong><span>{sourceMode === 'vdo' ? '貼上手機的 VIEW 觀看連結後連接' : '按下開啟後，才會請求使用攝影機'}</span></div>}
          {sourceMode === 'vdo' && cameraStatus === 'opening' && <div className="live-vdo-connecting"><LoaderCircle className="spin" size={14}/><span>等待手機影像，最長 30 秒</span></div>}
        </div>
        <div className="live-source-controls">
          {sourceMode === 'camera' ? <><div className="live-device-select"><label className="field"><span>攝影機裝置</span><select value={deviceId} disabled={active || cameraStatus === 'opening'} onChange={event => setDeviceId(event.target.value)}><option value="">瀏覽器預設攝影機</option>{devices.map((device, index) => <option key={device.deviceId || index} value={device.deviceId}>{device.label || `攝影機 ${index + 1}（開啟權限後顯示名稱）`}</option>)}</select></label><button className="icon-button" title="重新整理攝影機裝置" aria-label="重新整理攝影機裝置" disabled={active || cameraStatus === 'opening'} onClick={listDevices}><RefreshCw size={16}/></button></div>
          <button className="button secondary full" disabled={active || cameraStatus === 'opening'} onClick={openCamera}>{cameraStatus === 'opening' ? <LoaderCircle className="spin" size={15}/> : <Camera size={15}/>} {cameraStatus === 'ready' ? '重新開啟選取的攝影機' : '開啟攝影機'}</button>
          <label className="check-field"><input type="checkbox" checked={mirror} onChange={event => setMirror(event.target.checked)}/><span>鏡像預覽 <small>僅調整本機預覽方向</small></span></label></> : <>
          <label className="field live-vdo-url"><span>VDO.Ninja VIEW 觀看連結</span><input type="url" autoComplete="off" spellCheck="false" placeholder="https://vdo.ninja/?view=…" value={vdoUrl} disabled={active || cameraStatus === 'opening'} onChange={event => setVdoUrl(event.target.value)} onKeyDown={event => { if (event.key === 'Enter') connectVdo(); }}/></label>
          <label className="field live-vdo-method"><span>接收方式</span><select value={vdoMode} disabled={active || cameraStatus === 'opening'} onChange={event => changeVdoMode(event.target.value)}><option value="direct">直接接收</option><option value="iframe">相容模式（iframe）</option></select></label>
          <button className="button secondary full" disabled={active || cameraStatus === 'opening' || !vdoUrl.trim()} onClick={connectVdo}>{cameraStatus === 'opening' ? <LoaderCircle className="spin" size={15}/> : <Wifi size={15}/>} {cameraStatus === 'opening' ? '正在連接 VDO.Ninja' : cameraStatus === 'ready' ? '重新連接 VDO.Ninja' : '連接 VDO.Ninja'}</button>
          <p className="live-source-note">直接接收手機分享的影像。請使用含有 <code>view=</code> 的觀看連結，並保持手機分享頁面開啟。</p>
          <div className="live-receiver-stats"><span>來源接收 <strong>{Number.isFinite(receiverStats?.fps) ? `${receiverStats.fps.toFixed(1)} FPS` : '—'}</strong></span><span>來源解析度 <strong>{receiverStats?.width > 0 && receiverStats?.height > 0 ? `${receiverStats.width} × ${receiverStats.height}` : '—'}</strong></span>{Number.isFinite(receiverStats?.bitrateKbps) && <small>接收位元率 {receiverStats.bitrateKbps.toFixed(0)} Kbps</small>}{vdoMode === 'iframe' && <small>相容模式未提供來源接收 FPS。</small>}</div>
          </>}
          <div className="live-capture-settings"><label className="field"><span>目標傳送 FPS</span><select value={captureFps} disabled={active} onChange={event => setCaptureFps(Number(event.target.value))}>{[1, 2, 4, 8, 10, 15, 24, 25, 30].map(value => <option value={value} key={value}>{value} FPS</option>)}</select></label><div title="最近約 4 秒內，實際送達工作台的影格數／秒"><span>實際傳送</span><strong>{captureStats.fps.toFixed(1)} <small>FPS</small></strong></div></div>
          <p className="live-source-note">影格最高 960 × 720。目標 FPS 是傳送上限；實際傳送與辨識更新分別量測。影格轉換、大型模型及多模型比較都可能降低速度。</p>
          {sourceMode === 'vdo' && <p className="live-source-note">VDO.Ninja 預覽的 FPS 代表串流播放速度；模型辨識速度請看右側各區塊的「辨識更新」。</p>}
          <div className="live-pipeline-diagnostics"><div className="live-diagnostics-heading"><strong>傳送耗時</strong><span>{cameraStatus === 'ready' && active && !stopping ? '最近 4 秒平均' : diagnostics.sample_count ? '停止前最後量測' : '等待傳送'}</span></div><div className="live-diagnostics-grid"><span title="相容模式中，請求 VDO.Ninja 截圖到收到影格的時間；直接接收沒有此階段">取得影格 <b>{measuredMs(diagnostics.mean.capture_wait_ms)}</b></span><span title="影格解碼、縮放與 JPEG 轉換時間">轉換 JPEG <b>{measuredMs(diagnostics.mean.convert_ms)}</b></span><span title="送出影格到收到本機服務確認的時間，包含伺服器處理">送達本機 <b>{measuredMs(diagnostics.mean.upload_ms)}</b></span></div><div className="live-diagnostics-details"><span title="從開始取得此影格到本機確認收件，包含等待傳送的時間">影格總耗時 <b>{measuredMs(diagnostics.mean.cycle_ms)}</b></span><span>排程延遲 <b>{measuredMs(diagnostics.mean.timer_lag_ms)}</b></span><span>每張 JPEG <b>{Number.isFinite(diagnostics.mean.jpeg_bytes) ? `${(diagnostics.mean.jpeg_bytes / 1024).toFixed(0)} KB` : '—'}</b></span><span>頁面：<b>{pageVisibility === 'visible' ? '前景' : pageVisibility === 'hidden' ? '背景' : '未知'}</b></span>{Number.isFinite(diagnostics.mean.source_bytes) && <span>來源 PNG <b>{(diagnostics.mean.source_bytes / 1024).toFixed(0)} KB</b></span>}</div>{!diagnostics.sample_count && <p>開始傳送後顯示實測耗時。</p>}</div>
          <div className="live-duration-settings">
            <label className="field"><span>時間限制</span><select value={durationMode} disabled={active} onChange={event => setDurationMode(event.target.value)}><option value="custom">自訂時間</option><option value="unlimited">不設限</option></select></label>
            {durationMode === 'custom' ? <div className="live-duration-inputs"><label className="field"><span>自訂時間</span><input type="number" min="1" step="1" value={durationValue} disabled={active} aria-invalid={Boolean(duration.error)} aria-describedby={duration.error ? 'live-duration-error' : undefined} onChange={event => setDurationValue(event.target.value)}/></label><label className="field"><span>時間單位</span><select value={durationUnit} disabled={active} onChange={event => setDurationUnit(event.target.value)}><option value="seconds">秒</option><option value="minutes">分鐘</option><option value="hours">小時</option></select></label></div> : <p className="live-duration-hint">持續執行，直到手動停止、離開此頁或影像連線中斷。</p>}
            {duration.error && <p id="live-duration-error" className="live-duration-error" role="alert">{duration.error}</p>}
          </div>
          <div className="live-session-status"><i className={active ? 'active' : ''}/><span>{sessionLabel}<small>時間限制：{session ? formatLiveDuration(session.max_duration_seconds) : duration.label || '請輸入有效時間'}</small></span></div>
          <button className="button primary full" disabled={active || cameraStatus !== 'ready' || !cards.length || licenseMissing || checkpointsMissing || unavailableDevice || weightsMissing || Boolean(duration.error)} title={duration.error || (weightsMissing ? '請選擇可用的原始或微調權重' : '') || (unavailableDevice ? '選取的運算裝置目前不可用' : '') || (checkpointsMissing ? '選取的權重已不可用，請重新選擇' : '') || (licenseMissing ? '請先閱讀 RF-DETR 授權並勾選確認' : '')} onClick={startLive}>{starting ? <LoaderCircle className="spin" size={15}/> : <Play size={14} fill="currentColor"/>}開始即時辨識</button>
          <button className="button secondary full" disabled={cameraStatus === 'idle' && !active || stopping} onClick={() => stopAll()}><Square size={13}/>{sourceMode === 'vdo' ? '停止並中斷連線' : '停止並關閉攝影機'}</button>
          {weightsMissing && !checkpointsMissing && <p className="live-license-notice">請為每個區塊選擇可用的原始或微調權重。</p>}{unavailableDevice && <p className="live-license-notice">選取的運算裝置目前不可用，請展開模型參數重新選擇。</p>}{checkpointsMissing && <p className="live-license-notice">選取的權重已不可用，請重新選擇或在模型與任務檢查路徑。</p>}{licenseMissing && <p className="live-license-notice">RF-DETR 需要先閱讀授權並勾選確認。</p>}
          <p className="live-source-note">離開此頁會停止影像來源；即時辨識期間，其他推論與訓練會等待。</p>
          <p className="live-source-note">不錄影；保留最後辨識畫面與任務摘要，暫存原始影格在停止後清除。</p>
          {session?.id && <small className="live-session-id">任務 {session.id.slice(0, 8)} · {finished ? statusNames[session.status] : `${captureStats.count} 個影格已送出`}</small>}
        </div>
      </aside>
      <section className="live-comparison"><div className="live-comparison-heading"><div><h2>即時比較 <span>{cards.length} / 4</span></h2><p>每個區塊可選不同模型、參數或微調權重。調整設定前請先停止。</p></div><button className="button primary small" disabled={active || cards.length >= 4 || !availableModels.length} onClick={addCard}><Plus size={15}/>新增區塊</button></div>
        {error && <div className="live-error" role="alert"><CircleHelp size={17}/><span>{error}</span></div>}
        {finished && <div className="live-finished-note">{session.status === 'failed' ? '即時任務失敗。' : '即時任務已結束。'}以下保留本次最後一次收到的結果，並非即時影像。</div>}
        <div className="live-model-grid">{cards.map((card, index) => <ModelCard key={card.id} card={card} index={index} models={availableModels} jobs={jobs} registeredCheckpoints={checkpoints} confirmLicense={confirmLicense} disabled={active} result={liveState?.models?.find(result => String(result.key) === String(index))} finished={finished} onChange={patch => updateCard(card.id, patch)} onRemove={() => { setCards(previous => previous.filter(item => item.id !== card.id)); setLiveState(null); setSession(null); }}/>)}</div>
        {!cards.length && <div className="live-empty"><Monitor size={32}/><h3>新增一個即時比較區塊</h3><p>先從 YOLO26 開始，再加入其他模型觀察速度與效果。</p></div>}
      </section>
    </div><canvas ref={canvasRef} hidden/>
  </main>;
}
