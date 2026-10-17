import test from 'node:test';
import assert from 'node:assert/strict';
import { JobPoller, jobPollingDelay, mergeJobSnapshot, mergeJobUpdate } from '../src/jobPolling.js';

const job = (id, status, kind = 'inference') => ({ id, status, kind });
const deferred = () => { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; };
const flush = async () => { for (let i = 0; i < 5; i++) await Promise.resolve(); };

function harness(initial = []) {
  let jobs = initial, clock = 0, timerId = 0;
  const timers = new Map(), requests = [], errors = [], snapshots = [];
  const poller = new JobPoller({
    fetchJobs: signal => { const request = { ...deferred(), signal, at: clock }; requests.push(request); return request.promise; },
    getJobs: () => jobs,
    onJobs: incoming => { jobs = mergeJobSnapshot(jobs, incoming); snapshots.push(jobs); },
    onError: error => errors.push(error.message),
    setTimer: (callback, delay) => { const id = ++timerId; timers.set(id, { callback, at: clock + delay }); return id; },
    clearTimer: id => timers.delete(id),
  });
  return { poller, requests, timers, errors, snapshots,
    get jobs() { return jobs; },
    update(value) { jobs = mergeJobUpdate(jobs, value); poller.wake(); },
    async advance(ms) {
      const end = clock + ms;
      while (true) {
        const next = [...timers].sort((a, b) => a[1].at - b[1].at)[0];
        if (!next || next[1].at > end) break;
        clock = next[1].at; timers.delete(next[0]); next[1].callback(); await flush();
      }
      clock = end; await flush();
    },
  };
}

test('only queued or running inference uses the fast cadence', () => {
  assert.equal(jobPollingDelay([]), 2200);
  assert.equal(jobPollingDelay([job('a', 'running', 'training'), job('b', 'running', 'live')]), 2200);
  for (const status of ['queued', 'running']) assert.equal(jobPollingDelay([job('a', status)]), 225);
  for (const status of ['completed', 'failed', 'cancelled']) assert.equal(jobPollingDelay([job('a', status)]), 2200);
});

test('submission wakes idle polling immediately, polls at 225 ms, and completion restores low frequency', async () => {
  const h = harness(); h.poller.start(); await h.advance(0);
  h.requests[0].resolve([]); await flush();
  await h.advance(100); h.update(job('a', 'queued')); await h.advance(0);
  assert.equal(h.requests[1].at, 100);
  h.requests[1].resolve([job('a', 'running')]); await flush();
  await h.advance(224); assert.equal(h.requests.length, 2);
  await h.advance(1); assert.equal(h.requests[2].at, 325);
  h.requests[2].resolve([job('a', 'completed')]); await flush();
  await h.advance(2199); assert.equal(h.requests.length, 3);
  await h.advance(1); assert.equal(h.requests[3].at, 2525);
  h.poller.stop(); h.requests[3].resolve([]); await flush();
});

test('a slow pre-submission response is cancelled, ignored, and never overlaps another jobs request', async () => {
  const h = harness(); h.poller.start(); await h.advance(0);
  h.update(job('new', 'queued')); await h.advance(5000);
  assert.equal(h.requests[0].signal.aborted, true);
  assert.equal(h.requests.length, 1);
  assert.equal(h.snapshots.length, 0);
  h.requests[0].resolve([]); await flush(); await h.advance(0);
  assert.equal(h.requests.length, 2);
  assert.deepEqual(h.jobs, [job('new', 'queued')]);
  h.requests[1].resolve([job('new', 'completed')]); await flush();
  assert.equal(h.jobs[0].status, 'completed');
  h.poller.stop();
});

test('late submit and snapshot data cannot regress running or terminal jobs or remove a newly submitted job', () => {
  const completed = { ...job('done', 'completed'), result: { inference_ms: 42 } };
  const current = [job('new', 'queued'), completed, job('running', 'running')];
  const merged = mergeJobSnapshot(current, [job('done', 'running'), job('running', 'queued')]);
  assert.deepEqual(merged, current);
  assert.deepEqual(mergeJobUpdate(current, job('done', 'queued')), current);
  for (const status of ['completed', 'failed', 'cancelled']) {
    const terminal = job('a', status);
    assert.deepEqual(mergeJobUpdate([terminal], job('a', 'running')), [terminal]);
  }
});

test('cancellation invalidates old snapshots and abort errors do not become connection failures', async () => {
  const h = harness([job('a', 'running')]); h.poller.start(); await h.advance(0);
  h.update(job('a', 'cancelled'));
  h.requests[0].reject(new DOMException('cancelled request', 'AbortError'));
  await flush(); await h.advance(0);
  assert.deepEqual(h.errors, []);
  assert.equal(h.requests.length, 2);
  h.requests[1].resolve([job('a', 'running')]); await flush();
  assert.equal(h.jobs[0].status, 'cancelled');
  h.poller.stop();
});

test('stop clears timers, aborts the request and suppresses late responses and failures', async () => {
  for (const failure of [false, true]) {
    const h = harness([job('a', 'running')]); h.poller.start(); await h.advance(0);
    h.poller.stop();
    assert.equal(h.requests[0].signal.aborted, true);
    if (failure) h.requests[0].reject(new Error('late failure'));
    else h.requests[0].resolve([job('a', 'completed')]);
    await flush(); await h.advance(10000);
    assert.equal(h.requests.length, 1);
    assert.equal(h.timers.size, 0);
    assert.deepEqual(h.snapshots, []);
    assert.deepEqual(h.errors, []);
  }
});

test('network failure retries serially while invalidated errors stay silent', async () => {
  const h = harness(); h.poller.start(); await h.advance(0);
  h.requests[0].reject(new Error('offline')); await flush();
  assert.deepEqual(h.errors, ['offline']);
  await h.advance(2200); assert.equal(h.requests.length, 2);
  h.poller.wake(); h.requests[1].reject(new Error('superseded error')); await flush(); await h.advance(0);
  assert.deepEqual(h.errors, ['offline']);
  assert.equal(h.requests.length, 3);
  h.poller.stop(); h.requests[2].resolve([]); await flush();
});
