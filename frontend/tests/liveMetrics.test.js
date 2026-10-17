import test from 'node:test';
import assert from 'node:assert/strict';
import { advanceFrameRate, livePollInterval } from '../src/liveMetrics.js';

test('steady 30 FPS uses completed counts and elapsed time, independently of poll frequency', () => {
  let frequent, sparse;
  for (let time = 0; time <= 10000; time += 100) frequent = advanceFrameRate(frequent, time * 30 / 1000, time);
  for (let time = 0; time <= 10000; time += 500) sparse = advanceFrameRate(sparse, time * 30 / 1000, time);
  assert.equal(frequent.rate, 30);
  assert.equal(sparse.rate, 30);
  assert.ok(frequent.samples.length <= 42);
});

test('a model finishing once every two seconds reports 0.5 FPS without zero/one oscillation', () => {
  let state;
  for (let time = 0; time <= 10000; time += 100) {
    state = advanceFrameRate(state, Math.floor(time / 2000), time);
    if (time >= 4000) assert.equal(state.rate, 0.5);
  }
});

test('a stalled completed counter decays to zero and records the age of its last change', () => {
  let state;
  for (let time = 0; time <= 5000; time += 100) state = advanceFrameRate(state, time * 30 / 1000, time);
  for (let time = 5100; time <= 10000; time += 100) state = advanceFrameRate(state, 150, time);
  assert.equal(state.rate, 0);
  assert.equal(state.ageSeconds, 5);
});

test('a counter reset or insufficient observation has no fabricated FPS', () => {
  let state = advanceFrameRate(null, 0, 0);
  assert.equal(state.rate, undefined);
  state = advanceFrameRate(state, 10, 500);
  assert.equal(state.rate, undefined);
  state = advanceFrameRate(state, 20, 1000);
  assert.equal(state.rate, 20);
  state = advanceFrameRate(state, 0, 1200);
  assert.equal(state.rate, undefined);
  assert.equal(state.samples.length, 1);
  assert.equal(advanceFrameRate(state, NaN, 1300), null);
});

test('delayed polls use the whole measured interval and duplicate times remain bounded', () => {
  let state = advanceFrameRate(null, 0, 0);
  state = advanceFrameRate(state, 30, 6000);
  assert.equal(state.rate, 5);
  state = advanceFrameRate(state, 30, 6000);
  assert.equal(state.samples.length, 2);
  assert.equal(state.rate, 5);
});

test('live result polling follows target FPS within the 2–30 requests per second bounds', () => {
  assert.equal(livePollInterval(30), 1000 / 30);
  assert.equal(livePollInterval(24), 1000 / 24);
  assert.equal(livePollInterval(8), 125);
  assert.equal(livePollInterval(1), 500);
  assert.equal(livePollInterval(100), 1000 / 30);
  assert.equal(livePollInterval(NaN), 500);
});
