export function classAliases(rows = []) {
  const aliases = {};
  for (const row of rows) {
    const source = String(row.source || '').trim(), target = String(row.target || '').trim();
    if (!source || !target) throw new Error('類別對照的「新資料類別」與「原有類別」都必須填寫。');
    if (Object.hasOwn(aliases, source)) throw new Error(`新資料類別 ${source} 重複設定對照。`);
    // Define an own property even for a legitimate class named __proto__.
    Object.defineProperty(aliases, source, { value: target, enumerable: true, configurable: true });
  }
  return aliases;
}

export function trainingRequest({ modelId, datasetId, checkpointId, mode = 'standard', replayDatasetId, aliases = {}, params = {} }) {
  return { kind: 'training', model_id: modelId, dataset_id: datasetId, params,
    ...(checkpointId ? { checkpoint_id: checkpointId } : {}), training_mode: mode,
    ...(mode === 'expand_classes' ? { replay_dataset_id: replayDatasetId || null, class_aliases: aliases } : {}) };
}

export function canTrainWithWeights(model, checkpointId, choices = []) {
  if (!model?.training_supported) return false;
  if (!checkpointId) return Boolean(model.available);
  return Boolean((model.available || model.environment_available) && choices.some(item => item.id === checkpointId && item.valid));
}

export function freshExpansionPreview(preview, signature) {
  return Boolean(preview?.signature === signature && preview?.result?.valid === true);
}

export function retentionRows(retention) {
  const key = retention?.metric;
  if (!['accuracy', 'map50_95'].includes(key)) return [];
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const before = retention.before?.old?.[key], oldAfter = retention.after?.old?.[key], newAfter = retention.after?.new?.[key];
  return [
    { label: '舊任務', before: finite(before) ? before : null, after: finite(oldAfter) ? oldAfter : null,
      delta: finite(before) && finite(oldAfter) ? oldAfter - before : null },
    { label: '新任務', before: null, after: finite(newAfter) ? newAfter : null, delta: null },
  ];
}
