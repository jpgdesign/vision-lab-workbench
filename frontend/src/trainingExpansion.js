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

export const trainingModes = {
  standard: { label: '標準微調 · 使用資料集的類別', description: '輸出類別依這次資料集建立。未在資料集中的原有類別不會自動保留。' },
  expand_frozen: { label: '凍結骨幹後擴充 · 免舊資料，同一模型', description: '固定骨幹、保留並擴充分類輸出，只用新資料訓練。原分類頭仍會更新，可能遺忘；沒有舊驗證資料時，舊能力尚未測量。' },
  expand_classes: { label: '混合新舊資料擴充 · 需要舊資料', description: '保留原有分類輸出、追加新類別，混合新舊訓練資料，再比較舊任務訓練前後的驗證結果。仍可能退步，請以實際指標判斷。' },
  dual_branch: { label: '固定原模型＋新增分支 · 免舊資料，兩分支推論', description: '完整保留原模型，另訓練新任務分支。推論依序執行兩個分支並分組顯示結果，增加儲存與處理時間；各分支分數是獨立的。' },
};

export const requiresReplay = mode => mode === 'expand_classes';
export const isExpansionMode = mode => ['expand_frozen', 'expand_classes', 'dual_branch'].includes(mode);
export const inferenceOnlyCheckpoint = entry => (entry?.training?.checkpoint_format || entry?.checkpoint_format) === 'frozen_branch_bundle';

export function effectiveTrainingParams(mode, task, params) {
  return mode === 'expand_frozen' ? { ...params, strategy: task === 'detection' ? 'freeze_backbone' : 'head_only',
    ...(task === 'detection' && Object.hasOwn(params, 'freeze') ? { freeze: 0 } : {}) } : params;
}

export function trainingRequest({ modelId, datasetId, checkpointId, mode = 'standard', replayDatasetId, aliases = {}, baseClassNames = [], params = {} }) {
  return { kind: 'training', model_id: modelId, dataset_id: datasetId, params,
    ...(checkpointId ? { checkpoint_id: checkpointId } : {}), training_mode: mode,
    ...(isExpansionMode(mode) ? { replay_dataset_id: replayDatasetId || null, class_aliases: aliases,
      ...(baseClassNames.length ? { base_class_names: baseClassNames } : {}) } : {}) };
}

export function canTrainWithWeights(model, checkpointId, choices = []) {
  if (!model?.training_supported) return false;
  if (!checkpointId) return Boolean(model.available);
  return Boolean((model.available || model.environment_available) && choices.some(item => item.id === checkpointId && item.valid && !inferenceOnlyCheckpoint(item)));
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
