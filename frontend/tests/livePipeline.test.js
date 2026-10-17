import test from 'node:test';
import assert from 'node:assert/strict';
import { LatestFramePipeline } from '../src/livePipeline.js';

const flush = async () => { for (let index = 0; index < 12; index += 1) await Promise.resolve(); };
class Clock {
  time = 0; nextId = 0; timers = new Map();
  now = () => this.time;
  setTimer = (callback, delay) => { const id = ++this.nextId; this.timers.set(id, { callback, at: this.time + Math.max(0, delay) }); return id; };
  clearTimer = id => this.timers.delete(id);
  sleep = delay => new Promise(resolve => this.setTimer(resolve, delay));
  async advance(delay) {
    const end = this.time + delay;
    await flush();
    for (;;) {
      const next = [...this.timers].filter(([, timer]) => timer.at <= end).sort((a, b) => a[1].at - b[1].at || a[0] - b[0])[0];
      if (!next) break;
      this.time = next[1].at; this.timers.delete(next[0]); next[1].callback(); await flush();
    }
    this.time = end; await flush();
  }
}
const create = (clock, options) => new LatestFramePipeline({ fps: 30, now: clock.now, setTimer: clock.setTimer, clearTimer: clock.clearTimer, ...options });

test('default browser timer wrappers never bind native timers to the pipeline instance', async t => {
  const clock = new Clock(); let accepted = 0;
  t.mock.method(globalThis, 'setTimeout', function (callback, delay) {
    assert.ok(this === undefined || this === globalThis, 'native Window timer received an invalid receiver');
    return clock.setTimer(callback, delay);
  });
  t.mock.method(globalThis, 'clearTimeout', function (id) {
    assert.ok(this === undefined || this === globalThis, 'native Window timer received an invalid receiver');
    clock.clearTimer(id);
  });
  const pipeline = new LatestFramePipeline({ fps: 30, now: clock.now, capture: () => ({}), upload: () => {},
    onAccepted: () => { accepted += 1; }, onError: error => assert.fail(error.message) });
  pipeline.start(); await clock.advance(100); pipeline.stop();
  assert.ok(accepted >= 3);
});

test('capture and upload overlap but each stage has at most one operation in flight', async () => {
  const clock = new Clock(), captureStarts = [], uploadCompletes = [], accepted = [];
  let capturing = 0, uploading = 0, maxCapture = 0, maxUpload = 0, id = 0;
  const pipeline = create(clock, {
    capture: async () => { const value = ++id; capturing += 1; maxCapture = Math.max(maxCapture, capturing); captureStarts.push(clock.time); await clock.sleep(20); capturing -= 1; return value; },
    upload: async () => { uploading += 1; maxUpload = Math.max(maxUpload, uploading); await clock.sleep(20); uploading -= 1; uploadCompletes.push(clock.time); },
    onAccepted: value => accepted.push(value),
  });
  pipeline.start(); await clock.advance(1000); pipeline.stop();
  assert.equal(maxCapture, 1); assert.equal(maxUpload, 1);
  assert.ok(captureStarts[1] < uploadCompletes[0]);
  assert.ok(accepted.length >= 28 && accepted.length <= 30, accepted.length);
});

test('a slow upload retains only the newest waiting frame instead of building a queue', async () => {
  const clock = new Clock(), sent = []; let captured = 0, uploading = 0;
  const pipeline = create(clock, {
    capture: async () => { const id = ++captured; await clock.sleep(5); return id; },
    upload: async id => { uploading += 1; assert.equal(uploading, 1); sent.push(id); await clock.sleep(120); uploading -= 1; },
  });
  pipeline.start(); await clock.advance(500);
  assert.deepEqual(sent.slice(0, 2), [1, 4]);
  assert.ok(pipeline.getState().dropped > 0);
  assert.ok(sent.length < captured / 2);
  pipeline.stop(); assert.equal(pipeline.getState().hasPending, false);
});

test('absolute rounded-up cadence never runs a 33 ms loop that exceeds 30 FPS', async () => {
  const clock = new Clock(), captures = [], uploads = [];
  const pipeline = create(clock, { capture: () => { captures.push(clock.time); return {}; }, upload: () => { uploads.push(clock.time); } });
  pipeline.start(); await clock.advance(10000); pipeline.stop();
  for (let second = 0; second < 10; second += 1) {
    assert.equal(captures.filter(time => time >= second * 1000 && time < (second + 1) * 1000).length, 30);
    assert.ok(uploads.filter(time => time >= second * 1000 && time < (second + 1) * 1000).length <= 30);
  }
  captures.forEach((time, index) => assert.ok(time + 1e-8 >= index * 1000 / 30));
});

test('stopping during capture ignores its late result and never starts an upload', async () => {
  const clock = new Clock(); let uploads = 0, errors = 0;
  const pipeline = create(clock, { capture: async () => { await clock.sleep(100); return {}; }, upload: () => { uploads += 1; }, onError: () => { errors += 1; } });
  pipeline.start(); await clock.advance(10); pipeline.stop(); pipeline.stop(); await clock.advance(200);
  assert.equal(uploads, 0); assert.equal(errors, 0); assert.equal(pipeline.getState().hasPending, false);
});

test('stop aborts the current upload and suppresses all late acceptance callbacks', async () => {
  const clock = new Clock(); let signal, accepted = 0, errors = 0;
  const pipeline = create(clock, { capture: () => ({}), upload: async (value, currentSignal) => { signal = currentSignal; await clock.sleep(100); }, onAccepted: () => { accepted += 1; }, onError: () => { errors += 1; } });
  pipeline.start(); await clock.advance(5); pipeline.stop(); await clock.advance(200);
  assert.equal(signal.aborted, true); assert.equal(accepted, 0); assert.equal(errors, 0);
});

test('429 backs off both stages, discards stale frames, and resumes with a fresh capture', async () => {
  const clock = new Clock(), starts = [], sent = []; let id = 0, failures = 0;
  const pipeline = create(clock, {
    capture: async () => { starts.push(clock.time); const value = ++id; await clock.sleep(5); return value; },
    upload: async value => { sent.push({ value, at: clock.time }); await clock.sleep(1); if (sent.length === 1) throw Object.assign(new Error('rate limit'), { status: 429, retryAfter: '1' }); },
    onError: () => { failures += 1; },
  });
  pipeline.start(); await clock.advance(1200); pipeline.stop();
  assert.ok(starts[1] >= 1006, starts[1]);
  assert.ok(sent[1].at >= 1006); assert.ok(sent[1].value > sent[0].value);
  assert.equal(failures, 0);
});

test('upload timeout stops the pipeline even if the transport ignores abort', async () => {
  const clock = new Clock(); let signal; const errors = [];
  const pipeline = create(clock, { timeoutMs: 50, capture: () => ({}), upload: (value, currentSignal) => { signal = currentSignal; return new Promise(() => {}); }, onError: error => errors.push(error) });
  pipeline.start(); await clock.advance(100);
  assert.equal(signal.aborted, true); assert.equal(errors.length, 1); assert.equal(errors[0].name, 'TimeoutError');
  assert.equal(pipeline.getState().active, false); assert.equal(pipeline.getState().hasPending, false);
});

test('accepted frame timings retain capture, waiting, upload, and completion boundaries', async () => {
  const clock = new Clock(), timings = [];
  const pipeline = create(clock, { capture: async () => { await clock.sleep(12); return {}; }, upload: async () => { await clock.sleep(14); }, onAccepted: (value, timing) => timings.push(timing) });
  pipeline.start(); await clock.advance(30); pipeline.stop();
  assert.deepEqual(timings[0], { captureStartedAt: 0, captureCompletedAt: 12, uploadStartedAt: 12, completedAt: 26 });
});

test('immediate stop prevents capture work that was scheduled for the next microtask', async () => {
  const clock = new Clock(); let captures = 0, uploads = 0;
  const pipeline = create(clock, { capture: () => { captures += 1; return {}; }, upload: () => { uploads += 1; } });
  pipeline.start(); pipeline.stop(); await clock.advance(100);
  assert.equal(captures, 0); assert.equal(uploads, 0);
});

test('a capture started before rate backoff cannot be uploaded after it finally resolves', async () => {
  const clock = new Clock(), sent = []; let id = 0;
  const pipeline = create(clock, {
    capture: async () => { const value = ++id; await clock.sleep(value === 2 ? 1200 : 5); return value; },
    upload: async value => { sent.push(value); if (value === 1) { await clock.sleep(50); throw Object.assign(new Error('rate limit'), { status: 429, retryAfter: '1' }); } },
  });
  pipeline.start(); await clock.advance(1500); pipeline.stop();
  assert.equal(sent.includes(2), false);
  assert.ok(sent.some(value => value >= 3));
});
