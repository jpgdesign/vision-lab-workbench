import { modelDefaults } from './api.js';

export const modelFamilyKey = model => model?.family_id || model?.id || '';

export function modelFamilies(models = []) {
  const groups = new Map();
  for (const model of models) {
    const key = modelFamilyKey(model);
    if (!groups.has(key)) groups.set(key, { key, name: model.family_name || model.name, models: [] });
    groups.get(key).models.push(model);
  }
  return [...groups.values()];
}

export function chooseFamilyModel(models, family, preferredTask) {
  const variants = models.filter(model => modelFamilyKey(model) === family);
  const usable = model => model.available && model.inference_supported;
  return variants.find(model => model.task === preferredTask && usable(model))
    || variants.find(model => model.task === 'detection' && usable(model))
    || variants.find(model => model.task === preferredTask)
    || variants.find(model => model.task === 'detection')
    || variants[0];
}

// A task variant has its own architecture and weights. Clearing all checkpoint
// and result selectors prevents a detection card from retaining its old model.
export function switchedModelPatch(model) {
  if (!model) return null;
  return { modelId: model.id, params: modelDefaults(model), checkpointId: '', checkpointJobId: '', jobId: null, runMediaId: null };
}

export function modelReadiness(model) {
  if (!model?.inference_supported) return '推論尚未接入';
  if (!model.available) return '未就緒';
  return '';
}
