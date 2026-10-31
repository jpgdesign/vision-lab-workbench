import React from 'react';
import { applyGroundingTrainingTargets, groundingTrainingTargets } from './groundingTrainingTargets';

export default function GroundingTargetsControl({ modelId, checkpoint, params, disabled, onChange }) {
  const targets = groundingTrainingTargets(modelId, checkpoint);
  if (!targets) return null;
  const complete = params?.use_training_classes === true;
  const tooLong = !targets.chunked && targets.prompt.length > 2000;
  return <div className="grounding-training-targets">
    <label className="check-field"><input type="checkbox" checked={complete} disabled={disabled} onChange={event => onChange({ ...params, use_training_classes: event.target.checked })}/><span>使用完整微調類別</span></label>
    <p className="training-source-note">{targets.classCount > 0 ? `${targets.classCount} 個微調類別` : '已儲存的微調類別'} · {targets.chunkCount} 組偵測目標。{complete ? targets.chunked ? '依序處理全部分組後合併結果，推論時間會隨組數增加。' : '使用權重中儲存的完整類別文字。' : '目前使用自訂偵測目標；勾選後可辨識全部微調類別。'}</p>
    <button className="button ghost small" disabled={disabled || tooLong} title={tooLong ? '請勾選使用完整微調類別，或自行輸入需要辨識的類別。' : targets.chunked ? `使用全部 ${targets.classCount} 個類別、${targets.chunkCount} 組偵測目標` : targets.prompt} onClick={() => onChange(applyGroundingTrainingTargets(params, targets))}>套用訓練類別為偵測目標</button>
  </div>;
}
