import React, { useRef, useState } from 'react';
import { Check, CircleAlert, FileCheck2, LoaderCircle, Plus, Trash2, Upload } from 'lucide-react';
import { api } from './api';
import { retentionRows } from './trainingExpansion';
import './trainingExpansion.css';

const formatNames = { yolo_detection: 'YOLO Detection', coco_detection: 'COCO Detection', imagefolder: 'ImageFolder', video_csv: 'Video CSV' };
const message = value => typeof value === 'string' ? value : JSON.stringify(value);

export function RetentionSummary({ retention }) {
  if (!retention) return null;
  const rows = retentionRows(retention);
  const percent = value => value == null ? '未測量' : `${(value * 100).toFixed(2)}%`;
  return <section className="retention-summary" aria-label="新舊能力驗證結果">
    <h3>新舊能力驗證 <span>{retention.metric === 'accuracy' ? 'Accuracy' : 'mAP50–95'}</span></h3>
    <div className="retention-table-wrap"><table><thead><tr><th>任務</th><th>訓練前</th><th>最佳權重</th><th>差異</th></tr></thead><tbody>{rows.map(row => <tr key={row.label}><th>{row.label}</th><td>{percent(row.before)}</td><td>{percent(row.after)}</td><td className={row.delta != null && row.delta < 0 ? 'metric-decreased' : ''}>{row.delta == null ? '—' : `${row.delta >= 0 ? '+' : ''}${(row.delta * 100).toFixed(2)} pp`}</td></tr>)}</tbody></table></div>
    <p>{typeof retention.scope === 'string' ? retention.scope : '舊任務使用相同驗證資料比較訓練前與最佳權重；不代表所有情境的能力。'}</p>
    {retention.warnings?.length > 0 && <div className="validation-warnings">{retention.warnings.map((warning, index) => <p key={index}>{message(warning)}</p>)}</div>}
    {rows[0]?.before == null && rows[0]?.after == null && <p>舊任務尚未測量；這不表示分數為 0，也不能據此判定保留效果。</p>}
    <small>差異以百分點（pp）表示。凍結骨幹或混合舊資料不保證準確率完全不下降；雙分支模式保留原分支，分支分數須分開解讀。</small>
  </section>;
}

export function ReplayDataset({ datasets, setDatasets, formats = [], selectedId, onChange, onBusy, notify, required = true }) {
  const [format, setFormat] = useState(formats[0] || '');
  const [path, setPath] = useState(''), [loading, setLoading] = useState(false), [validation, setValidation] = useState(null);
  const upload = useRef(null);
  const effectiveFormat = formats.includes(format) ? format : formats[0] || '';
  const choices = datasets.filter(item => formats.includes(item.format));
  const selected = choices.find(item => item.id === selectedId);
  async function validate(body, endpoint) {
    setLoading(true); onBusy(true); setValidation(null); onChange('');
    try {
      const result = await api(endpoint, { method: 'POST', body }); setValidation(result);
      if (result.valid && result.dataset) {
        setDatasets(previous => [result.dataset, ...previous.filter(item => item.id !== result.dataset.id)]);
        onChange(result.dataset.id); notify('舊資料集已驗證並選用。');
      }
    } catch (error) { setValidation({ errors: [error.message] }); notify(error.message, true); }
    finally { setLoading(false); onBusy(false); }
  }
  return <section className="replay-dataset">
    <h3>{required ? '混合訓練的舊資料集（必填）' : '舊任務驗證資料（選填）'}</h3><p>{required ? '提供原有類別的訓練與驗證資料。新舊資料都參與訓練；驗證切分會各自評估。' : '可以不提供舊資料並直接訓練；提供後僅用驗證切分評估舊任務，不混入本次訓練。未提供時，舊能力指標會標示未測量。資料集仍需符合既有 train／val 格式。'}</p>
    <label className="field"><span>已驗證的舊資料集</span><select value={selectedId} onChange={event => { onChange(event.target.value); setValidation(null); }} disabled={loading}><option value="">{required ? '選擇要保留能力的資料集' : '不提供舊資料 · 舊能力未測量'}</option>{choices.map(item => <option key={item.id} value={item.id}>{item.name} · {formatNames[item.format]} · {item.train_count} train / {item.val_count} val</option>)}</select></label>
    {selected && <p className="replay-selection"><Check size={14}/>{selected.classes?.length || selected.num_classes} 類別 · {selected.train_count} 訓練／{selected.val_count} 驗證</p>}
    <details><summary>登錄其他舊資料集（本機路徑或 ZIP）</summary><div className="replay-import">
      <label className="field"><span>舊資料格式</span><select value={effectiveFormat} disabled={loading} onChange={event => { setFormat(event.target.value); setValidation(null); }}>{formats.map(value => <option key={value} value={value}>{formatNames[value]}</option>)}</select></label>
      <label className="field"><span>舊資料本機路徑</span><input value={path} onChange={event => setPath(event.target.value)} disabled={loading} placeholder={effectiveFormat === 'yolo_detection' ? 'dataset.yaml 的絕對路徑' : '資料集根資料夾的絕對路徑'}/></label>
      <div className="replay-import-buttons"><button className="button secondary small" disabled={loading || !path.trim() || !effectiveFormat} onClick={() => validate({ path: path.trim(), format: effectiveFormat }, '/datasets/validate')}>{loading ? <LoaderCircle className="spin" size={14}/> : <FileCheck2 size={14}/>}驗證舊資料</button><button className="button ghost small" disabled={loading || !effectiveFormat} onClick={() => upload.current?.click()}><Upload size={14}/>上傳舊資料 ZIP</button></div>
      <input type="file" hidden ref={upload} accept=".zip,application/zip" onChange={event => { const file = event.target.files?.[0]; event.target.value = ''; if (!file) return; const body = new FormData(); body.append('file', file); body.append('format', effectiveFormat); validate(body, '/datasets/upload'); }}/>
    </div></details>
    {validation?.errors?.length > 0 && <div className="validation-errors"><CircleAlert size={15}/><div>{validation.errors.map((error, index) => <p key={index}>{message(error)}</p>)}</div></div>}
    {validation?.warnings?.length > 0 && <div className="validation-warnings">{validation.warnings.map((warning, index) => <p key={index}>{message(warning)}</p>)}</div>}
  </section>;
}

export function ClassAliases({ rows, setRows, classes = [], baseClasses = [] }) {
  return <details className="class-aliases"><summary>類別名稱對照（選填）</summary>
    <p>同一概念使用不同名稱時，可明確對應到原有類別，例如將新資料的「person_new」對應到「person」。未對照的新名稱會追加為新類別。</p>
    {rows.map((row, index) => <div className="alias-row" key={index}><label className="field"><span>新資料類別</span><select value={row.source} onChange={event => setRows(previous => previous.map((value, i) => i === index ? { ...value, source: event.target.value } : value))}><option value="">選擇類別</option>{classes.map(name => <option key={name} value={name}>{name}</option>)}</select></label><span aria-hidden="true">→</span><label className="field"><span>原有類別名稱</span><input list="existing-training-classes" value={row.target} onChange={event => setRows(previous => previous.map((value, i) => i === index ? { ...value, target: event.target.value } : value))} placeholder="輸入完整原有類別名稱"/></label><button className="button ghost icon" aria-label={`刪除第 ${index + 1} 個類別對照`} onClick={() => setRows(previous => previous.filter((_, i) => i !== index))}><Trash2 size={15}/></button></div>)}
    <datalist id="existing-training-classes">{baseClasses.map(name => <option key={name} value={name}/>)}</datalist>
    <button className="button ghost small" disabled={!classes.length} onClick={() => setRows(previous => [...previous, { source: '', target: '' }])}><Plus size={14}/>新增類別對照</button>
  </details>;
}

export function ExpansionPreview({ result, mode }) {
  if (!result) return null;
  const plan = result.plan;
  const names = { old_train: '舊資料 · 訓練', old_val: '舊資料 · 驗證', new_train: '新資料 · 訓練', new_val: '新資料 · 驗證' };
  return <section className={`expansion-preview ${result.valid ? 'valid' : ''}`} aria-live="polite">
    <h3>{result.valid ? <Check size={16}/> : <CircleAlert size={16}/>} {result.valid ? '擴充計畫驗證通過' : '擴充計畫需要修正'}</h3>
    {result.errors?.length > 0 && <div className="validation-errors"><div>{result.errors.map((error, index) => <p key={index}>{message(error)}</p>)}</div></div>}
    {plan && <><div className="expansion-counts"><span><b>{plan.base_classes?.length ?? '—'}</b>原有類別</span><span><b>{plan.new_class_ids?.length ?? '—'}</b>追加類別</span><span><b>{plan.classes?.length ?? '—'}</b>{mode === 'dual_branch' ? '兩分支類別聯集' : '訓練總類別'}</span></div>
      <div className="coverage-grid">{Object.entries(plan.coverage || {}).map(([key, value]) => { const counts = value?.counts || {}, countValues = Object.values(counts), missingData = value?.provided === false || value?.measured === false; return <details key={key}><summary>{names[key] || key}<strong>{missingData ? '未提供 · 舊能力未測量' : `${countValues.filter(n => Number(n) > 0).length} / ${countValues.length} 類別有樣本`}</strong></summary>{missingData ? <p>沒有提供這份資料，不會產生對應的舊能力指標。</p> : value?.missing?.length ? <p className="metric-decreased">缺少：{value.missing.join('、')}</p> : <p>這個切分的目標類別都有樣本。</p>}{!missingData && <ul>{Object.entries(counts).map(([name, count]) => <li key={name}><span>{name}</span><b>{count}</b></li>)}</ul>}</details>; })}</div>
      <details className="union-classes"><summary>查看合併後類別順序</summary><p>{plan.classes?.map((name, index) => `${index}: ${name}`).join(' · ')}</p></details></>}
    {result.warnings?.length > 0 && <div className="validation-warnings">{result.warnings.map((warning, index) => <p key={index}>{message(warning)}</p>)}</div>}
  </section>;
}
