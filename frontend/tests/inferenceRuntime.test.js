import test from 'node:test';
import assert from 'node:assert/strict';
import { inferenceRuntimeDetails } from '../src/inferenceRuntime.js';

test('legacy results do not invent cache state or load and total timings', () => {
  for (const result of [undefined, null, {}, { inference_ms: 20 }, { runtime: {} }]) {
    assert.deepEqual(inferenceRuntimeDetails(result), { cacheLabel: null, loadTime: null, totalTime: null });
  }
});

test('cache hits preserve measured zero loading time and the separately measured total', () => {
  assert.deepEqual(inferenceRuntimeDetails({ inference_ms: 20, runtime: { model_cache_hit: true, model_load_seconds: 0, total_seconds: 0.032 } }), {
    cacheLabel: '重用已載入模型', loadTime: '0 ms', totalTime: '32 ms',
  });
});

test('cache misses display loading seconds independently of inference and end-to-end time', () => {
  assert.deepEqual(inferenceRuntimeDetails({ inference_ms: 20, runtime: { model_cache_hit: false, model_load_seconds: 2.125, total_seconds: 3.456 } }), {
    cacheLabel: '本次載入模型', loadTime: '2.13 s', totalTime: '3.46 s',
  });
});

test('malformed cache flags and nonnumeric, negative or nonfinite times stay absent', () => {
  for (const value of [undefined, null, '', '0', 'false', 1, false, NaN, Infinity, -1]) {
    const result = inferenceRuntimeDetails({ runtime: { model_cache_hit: value, model_load_seconds: value, total_seconds: value } });
    assert.equal(result.cacheLabel, value === false ? '本次載入模型' : null);
    assert.equal(result.loadTime, value === 1 ? '1.00 s' : null);
    assert.equal(result.totalTime, value === 1 ? '1.00 s' : null);
  }
});

test('partial measurements never derive missing totals or imply cache hits from zero time', () => {
  assert.deepEqual(inferenceRuntimeDetails({ inference_ms: 20, runtime: { model_load_seconds: 0 } }), {
    cacheLabel: null, loadTime: '0 ms', totalTime: null,
  });
  assert.deepEqual(inferenceRuntimeDetails({ runtime: { total_seconds: 1 } }), {
    cacheLabel: null, loadTime: null, totalTime: '1.00 s',
  });
});
