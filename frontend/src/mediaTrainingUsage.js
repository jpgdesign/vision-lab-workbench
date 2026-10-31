export function mediaTrainingUsage(media) {
  const used = media?.training_usage?.used === true;
  const jobIds = Array.isArray(media?.training_usage?.job_ids)
    ? [...new Set(media.training_usage.job_ids.filter(value => typeof value === 'string' && value))] : [];
  const name = typeof media?.name === 'string' ? media.name : '素材';
  return {
    used,
    accessibleLabel: used ? `${name} · 已用於訓練` : name,
    title: used ? `${name}\n曾用於本機已完成的訓練${jobIds.length ? ` · ${jobIds.length} 個工作` : ''}（不限目前選用模型）`
      : `${name}\n${media?.training_usage?.used === false ? '沒有符合的本機已完成訓練紀錄；不代表未曾用於其他訓練或基礎模型預訓練。' : '本機訓練使用紀錄尚未提供。'}`,
  };
}

export function completedTrainingSignature(jobs = []) {
  return JSON.stringify(jobs.filter(job => job.kind === 'training' && job.status === 'completed')
    .map(job => job.id).filter(id => typeof id === 'string').sort());
}

export function mergeMediaUsageSnapshot(previous, incoming) {
  const ids = new Set(incoming.map(item => item.id));
  // A user upload may finish after this refresh began. Keep new uploads that
  // were not yet visible to the server snapshot, while replacing known metadata.
  return [...incoming, ...previous.filter(item => !ids.has(item.id))];
}
