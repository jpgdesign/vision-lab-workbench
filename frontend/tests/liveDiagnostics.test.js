import test from 'node:test';
import assert from 'node:assert/strict';
import { recordPipelineSample, summarizePipeline } from '../src/liveDiagnostics.js';

test('stage averages and percentiles use measured numeric samples only', () => {
  let state;
  for (const [index, duration] of [10, 20, 30].entries()) {
    state = recordPipelineSample(state, { capture_wait_ms: duration, convert_ms: duration / 2, upload_ms: 4, jpeg_bytes: 20000, cycle_ms: duration * 1.5 + 4 }, index * 100);
  }
  const summary = summarizePipeline(state, 300);
  assert.equal(summary.sample_count, 3);
  assert.equal(summary.mean.capture_wait_ms, 20);
  assert.equal(summary.mean.convert_ms, 10);
  assert.equal(summary.mean.upload_ms, 4);
  assert.equal(summary.mean.cycle_ms, 34);
  assert.equal(summary.p95.capture_wait_ms, 30);
  assert.equal(summary.max.capture_wait_ms, 30);
  assert.equal(summary.mean.jpeg_bytes, 20000);
  assert.equal(summary.age_ms, 100);
});

test('missing stages remain absent instead of becoming fabricated zero timings', () => {
  let state = recordPipelineSample(null, { capture_wait_ms: 0 }, 0);
  state = recordPipelineSample(state, { upload_ms: 12 }, 100);
  const summary = summarizePipeline(state, 100);
  assert.equal(summary.sample_count, 2);
  assert.equal(summary.mean.capture_wait_ms, 0);
  assert.equal(summary.mean.upload_ms, 12);
  assert.equal(summary.counts.capture_wait_ms, 1);
  assert.equal(summary.counts.upload_ms, 1);
  assert.equal(summary.mean.convert_ms, undefined);
  assert.equal(summary.mean.jpeg_bytes, undefined);
});

test('unknown fields, source URLs, invalid numbers, and noninteger byte counts are discarded', () => {
  const state = recordPipelineSample(null, { capture_wait_ms: -1, convert_ms: Infinity, upload_ms: '2', jpeg_bytes: 12.5, timer_lag_ms: 0, source_url: 'private-view-url', image: new Uint8Array([1, 2]), arbitrary: 15 }, 50);
  assert.deepEqual(state.samples[0].values, { timer_lag_ms: 0 });
  assert.equal(JSON.stringify(state).includes('private-view-url'), false);
  const next = recordPipelineSample(state, { upload_ms: NaN }, 100);
  assert.equal(next.samples.length, 1);
  assert.equal(next.last_sample_at, 50);
});

test('stalled measurements expire after four seconds without becoming current zero values', () => {
  const state = recordPipelineSample(null, { upload_ms: 8 }, 100);
  assert.equal(summarizePipeline(state, 4099).sample_count, 1);
  const expired = summarizePipeline(state, 4100);
  assert.equal(expired.sample_count, 0);
  assert.deepEqual(expired.mean, {});
  assert.equal(expired.age_ms, 4000);
});

test('the diagnostic history is bounded even when many frames arrive inside the time window', () => {
  let state;
  for (let index = 0; index < 1000; index += 1) state = recordPipelineSample(state, { upload_ms: index }, index);
  assert.equal(state.samples.length, 240);
  assert.equal(state.samples[0].at, 760);
  state = recordPipelineSample(state, { upload_ms: 4 }, 6000);
  assert.equal(state.samples.length, 1);
  assert.equal(summarizePipeline(state, 6000).mean.upload_ms, 4);
});

test('clock or session reset cannot blend future measurements into a fresh capture', () => {
  let state = recordPipelineSample(null, { upload_ms: 100 }, 5000);
  state = recordPipelineSample(state, { upload_ms: 5 }, 10);
  assert.equal(state.samples.length, 1);
  assert.equal(summarizePipeline(state, 10).mean.upload_ms, 5);
  assert.equal(summarizePipeline(state, 0).sample_count, 0);
  assert.equal(summarizePipeline(state, 0).age_ms, null);
});
