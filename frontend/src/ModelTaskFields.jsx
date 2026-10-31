import React from 'react';
import { taskNames } from './api';
import { chooseFamilyModel, modelFamilies, modelFamilyKey, modelReadiness, switchedModelPatch } from './modelTasks';
import './modelTasks.css';

export default function ModelTaskFields({ models, modelId, onChange, disabled = false, label = '模型' }) {
  const model = models.find(item => item.id === modelId);
  const familyKey = modelFamilyKey(model);
  const families = modelFamilies(models);
  const variants = families.find(item => item.key === familyKey)?.models || [];
  function change(next) { const patch = switchedModelPatch(next); if (patch && next.id !== modelId) onChange(patch); }
  return <>
    <label className="field"><span>{label} <span className="tiny-label">MODEL</span></span><select disabled={disabled} value={familyKey} onChange={event => change(chooseFamilyModel(models, event.target.value, model?.task))}>
      {!model && <option value="">選擇模型</option>}
      {families.map(family => <option key={family.key} value={family.key}>{family.name}{family.models.some(item => item.available && item.inference_supported) ? '' : ' · 未就緒'}</option>)}
    </select></label>
    {variants.length > 1 && <label className="field model-task-field"><span>任務 <span className="tiny-label">TASK</span></span><select value={modelId} disabled={disabled} onChange={event => change(variants.find(item => item.id === event.target.value))}>
      {variants.map(item => <option key={item.id} value={item.id}>{taskNames[item.task] || item.task} · {item.name}{modelReadiness(item) ? ` · ${modelReadiness(item)}` : ''}</option>)}
    </select></label>}
    {variants.length > 1 && <p className="model-task-note">{model?.task === 'segmentation' ? '使用分割專用權重，輸出物件遮罩與輪廓。' : '使用偵測專用權重，輸出物件方框。'}切換任務會清除先前的權重選擇與結果。</p>}
    {model && !model.available && <p className="model-task-note unavailable">{model.reason || '目前模型權重或執行環境尚未就緒，請到模型與任務檢查。'}</p>}
  </>;
}
