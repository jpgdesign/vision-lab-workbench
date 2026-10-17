export function checkpointChoices(modelId, checkpoints = [], jobs = []) {
  const choices = new Map();
  for (const entry of checkpoints) {
    if (entry.model_id !== modelId || typeof entry.id !== 'string') continue;
    const job = entry.job_id ? jobs.find(item => item.id === entry.job_id) : null;
    choices.set(entry.id, { ...entry, valid: entry.valid !== false, name: entry.name || (entry.source === 'training_job' ? `工作台微調 · ${entry.job_id?.slice(0, 8) || entry.id}` : entry.id), training: entry.training || job?.result?.training });
  }
  for (const job of jobs) {
    const id = `job:${job.id}`;
    if (job.model_id !== modelId || job.kind !== 'training' || job.status !== 'completed' || !job.result?.checkpoint_url || choices.has(id)) continue;
    choices.set(id, { id, model_id: modelId, job_id: job.id, source: 'training_job', valid: true, name: `工作台微調 · ${new Date(job.created_at).toLocaleString('zh-TW', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })} · ${job.id.slice(0, 6)}`, training: job.result?.training });
  }
  return [...choices.values()];
}

export const selectedCheckpoint = data => data.checkpointId || (data.checkpointJobId ? `job:${data.checkpointJobId}` : '');

export function checkpointRequest(data) {
  if (data.checkpointId) return { checkpoint_id: data.checkpointId };
  return data.checkpointJobId ? { checkpoint_job_id: data.checkpointJobId } : {};
}

export function checkpointUnavailable(data, choices) {
  const id = selectedCheckpoint(data);
  if (!id) return false;
  return !choices.some(choice => choice.id === id && choice.valid);
}

export function canRunWithWeights(model, data, choices) {
  if (!model?.inference_supported) return false;
  if (!selectedCheckpoint(data)) return Boolean(model.available);
  return !checkpointUnavailable(data, choices) && Boolean(model.available || model.environment_available);
}
