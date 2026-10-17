function measuredDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return null;
  return seconds < 1 ? `${Math.round(seconds * 1000)} ms` : `${seconds.toFixed(2)} s`;
}

export function inferenceRuntimeDetails(result) {
  const runtime = result?.runtime;
  return {
    cacheLabel: runtime?.model_cache_hit === true ? '重用已載入模型' : runtime?.model_cache_hit === false ? '本次載入模型' : null,
    loadTime: measuredDuration(runtime?.model_load_seconds),
    totalTime: measuredDuration(runtime?.total_seconds),
  };
}
