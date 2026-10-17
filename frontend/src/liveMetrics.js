const WINDOW_MS = 4000;

// Polling observes completed counters. Averaging over four seconds keeps a
// slow model's rate steady without equating one response with one inference.
export function advanceFrameRate(previous, count, now) {
  if (!Number.isFinite(count) || count < 0 || !Number.isFinite(now)) return null;
  if (!previous || count < previous.count || now < previous.time) {
    return { samples: [{ count, time: now }], count, time: now, lastChangedAt: now, rate: undefined, ageSeconds: 0 };
  }
  const samples = [...previous.samples];
  const sample = { count, time: now };
  if (samples.at(-1)?.time === now) samples[samples.length - 1] = sample;
  else samples.push(sample);
  // Keep the sample immediately before the boundary so sparse polling still
  // measures the real elapsed interval instead of overstating throughput.
  while (samples.length > 1 && samples[1].time <= now - WINDOW_MS) samples.shift();
  const elapsed = now - samples[0].time;
  const lastChangedAt = count === previous.count ? previous.lastChangedAt : now;
  return {
    samples, count, time: now, lastChangedAt,
    rate: elapsed >= 1000 ? (count - samples[0].count) * 1000 / elapsed : undefined,
    ageSeconds: Math.floor((now - lastChangedAt) / 1000),
  };
}

export function livePollInterval(targetFps) {
  if (!Number.isFinite(targetFps) || targetFps <= 0) return 500;
  return Math.max(1000 / 30, Math.min(500, 1000 / targetFps));
}
