const WINDOW_MS = 4000;
const MAX_SAMPLES = 240;
const METRICS = ['capture_wait_ms', 'convert_ms', 'upload_ms', 'jpeg_bytes', 'source_bytes', 'cycle_ms', 'timer_lag_ms'];

// Numeric timings only. Source URLs, stream IDs, images, and other request
// fields never enter this in-memory diagnostic history.
export function recordPipelineSample(previous, sample, now = performance.now()) {
  if (!Number.isFinite(now) || now < 0) return previous || { samples: [], last_sample_at: null };
  const values = {};
  for (const name of METRICS) {
    const value = sample?.[name];
    if (!Number.isFinite(value) || value < 0) continue;
    if (name.endsWith('_bytes') && (!Number.isSafeInteger(value) || value < 1)) continue;
    values[name] = value;
  }
  const reset = previous?.last_sample_at != null && now < previous.last_sample_at;
  const samples = reset ? [] : (previous?.samples || []).filter(item => item.at > now - WINDOW_MS && item.at <= now);
  if (!Object.keys(values).length) return { samples, last_sample_at: reset ? null : previous?.last_sample_at ?? null };
  return { samples: [...samples, { at: now, values }].slice(-MAX_SAMPLES), last_sample_at: now };
}

export function summarizePipeline(state, now = performance.now()) {
  const validTime = Number.isFinite(now) && now >= 0;
  const samples = validTime ? (state?.samples || []).filter(item => item.at > now - WINDOW_MS && item.at <= now) : [];
  const summary = {
    sample_count: samples.length,
    age_ms: validTime && state?.last_sample_at != null && now >= state.last_sample_at ? now - state.last_sample_at : null,
    mean: {}, p95: {}, max: {}, counts: {},
  };
  for (const name of METRICS) {
    const values = samples.map(sample => sample.values[name]).filter(Number.isFinite).sort((a, b) => a - b);
    if (!values.length) continue;
    summary.counts[name] = values.length;
    summary.mean[name] = values.reduce((sum, value) => sum + value, 0) / values.length;
    summary.p95[name] = values[Math.ceil(values.length * 0.95) - 1];
    summary.max[name] = values.at(-1);
  }
  return summary;
}
