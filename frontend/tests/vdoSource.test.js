import test from 'node:test';
import assert from 'node:assert/strict';
import { parseVdoViewUrl, VdoFrameSource } from '../src/vdoSource.js';

class MessageTarget {
  listeners = new Set();
  addEventListener(type, listener) { assert.equal(type, 'message'); this.listeners.add(listener); }
  removeEventListener(type, listener) { assert.equal(type, 'message'); this.listeners.delete(listener); }
  emit(event) { return Promise.all([...this.listeners].map(listener => listener(event))); }
}

function harness(t, convert = async () => new Blob(['jpeg'], { type: 'image/jpeg' })) {
  const sent = [];
  const eventTarget = new MessageTarget();
  const contentWindow = { postMessage(data, origin) { sent.push({ data, origin }); } };
  const iframe = { contentWindow };
  const source = new VdoFrameSource({ iframe, source: parseVdoViewUrl('https://vdo.ninja/?view=phone_123'), eventTarget, convert });
  t.after(() => source.stop());
  return { source, sent, eventTarget, contentWindow };
}

function reply(h, patch = {}, valuePatch = {}) {
  return { origin: 'https://vdo.ninja', source: h.contentWindow,
    data: { action: 'image-frame-capture', value: { cib: h.sent.at(-1).data.cib,
      streamID: 'phone_123', imageData: new Blob(['png'], { type: 'image/png' }), ...valuePatch } }, ...patch };
}

test('viewer URL retains only secrets and forces safe viewer options', () => {
  const input = new URL('https://vdo.ninja/');
  const secrets = { password: 'a&b', pass: '中文密碼', pw: 'p w', p: '', salt: 's+1' };
  input.searchParams.set('view', 'phone_123-ABC');
  for (const [key, value] of Object.entries(secrets)) input.searchParams.set(key, value);
  for (const [key, value] of Object.entries({ api: '*', iframetarget: 'https://evil.example', url: 'https://evil.example/track', noaudio: 'false', muted: 'false', cleanoutput: 'false', unknownSecret: 'remove-me' })) {
    input.searchParams.set(key, value);
  }
  const parsed = parseVdoViewUrl(input.href, 'http://127.0.0.1:8765');
  const url = new URL(parsed.url);
  assert.equal(parsed.origin, 'https://vdo.ninja');
  assert.equal(parsed.streamId, 'phone_123-ABC');
  assert.equal(url.searchParams.get('iframetarget'), 'http://127.0.0.1:8765');
  for (const [key, value] of Object.entries(secrets)) assert.equal(url.searchParams.get(key), value);
  assert.deepEqual([...url.searchParams.keys()].sort(), ['view', ...Object.keys(secrets), 'autostart', 'cleanoutput', 'noaudio', 'muted', 'iframetarget'].sort());
  for (const key of ['autostart', 'cleanoutput', 'noaudio', 'muted']) assert.equal(url.searchParams.get(key), '');
});

test('viewer parser rejects nonofficial origins, paths, credentials, fragments and invalid IDs', () => {
  for (const value of ['', 'javascript:alert(1)', 'http://vdo.ninja/?view=id', 'https://vdo.ninja.evil.example/?view=id',
    'https://sub.vdo.ninja/?view=id', 'https://vdo.ninja:444/?view=id', 'https://u:p@vdo.ninja/?view=id',
    'https://vdo.ninja/tool?view=id', 'https://vdo.ninja/?view=id#settings', 'https://vdo.ninja/',
    'https://vdo.ninja/?view=', 'https://vdo.ninja/?view=a&view=b', 'https://vdo.ninja/?view=a%2Fb',
    `https://vdo.ninja/?view=${'a'.repeat(129)}`]) {
    assert.throws(() => parseVdoViewUrl(value), undefined, value);
  }
  assert.equal(parseVdoViewUrl('  https://vdo.ninja/?view=A_123-xyz  ').streamId, 'A_123-xyz');
});

test('viewer parser rejects publisher/director/room commands and ambiguous secret parameters', () => {
  for (const key of ['push', 'publish', 'director', 'room']) {
    assert.throws(() => parseVdoViewUrl(`https://vdo.ninja/?view=id&${key}=`), /VIEW/);
  }
  for (const key of ['password', 'pass', 'pw', 'p', 'salt']) {
    assert.throws(() => parseVdoViewUrl(`https://vdo.ninja/?view=id&${key}=a&${key}=b`), /密碼/);
    assert.throws(() => parseVdoViewUrl(`https://vdo.ninja/?view=id&${key}=${'x'.repeat(257)}`), /密碼/);
  }
});

test('one pending frame is shared, matching response resolves it and next request gets new correlation ID', async t => {
  let conversions = 0;
  const jpeg = new Blob(['jpeg'], { type: 'image/jpeg' });
  const h = harness(t, async () => { conversions += 1; return jpeg; });
  h.source.start();
  h.source.start();
  assert.equal(h.eventTarget.listeners.size, 1);
  const first = h.source.getFrame();
  assert.strictEqual(h.source.getFrame(), first);
  assert.equal(h.sent.length, 1);
  assert.deepEqual({ ...h.sent[0].data, cib: 'correlation' }, { getVideoFrame: true, streamID: 'phone_123', cib: 'correlation' });
  assert.equal(h.sent[0].origin, 'https://vdo.ninja');
  await h.eventTarget.emit(reply(h));
  assert.strictEqual(await first, jpeg);
  assert.equal(conversions, 1);
  assert.equal(h.source.pending, null);
  const next = h.source.getFrame();
  const rejected = assert.rejects(next, { name: 'AbortError' });
  assert.notEqual(h.sent[0].data.cib, h.sent[1].data.cib);
  h.source.stop();
  await rejected;
  assert.equal(h.eventTarget.listeners.size, 0);
});

test('only expected origin, iframe, action, request ID, stream ID and nonempty Blob are accepted', async t => {
  let conversions = 0;
  const h = harness(t, async () => { conversions += 1; return new Blob(['jpeg']); });
  h.source.start();
  const pending = h.source.getFrame();
  const rejected = assert.rejects(pending, { name: 'AbortError' });
  const invalid = [
    reply(h, { origin: 'https://evil.example' }),
    reply(h, { source: {} }),
    reply(h, { data: { action: 'other', value: reply(h).data.value } }),
    reply(h, {}, { cib: 'old-request' }),
    reply(h, {}, { streamID: 'different-stream' }),
    reply(h, {}, { imageData: new Blob([]) }),
    reply(h, {}, { imageData: { size: 10, type: 'image/png' } }),
    reply(h, { data: null }),
    reply(h, { data: { action: 'image-frame-capture', value: null } }),
  ];
  for (const event of invalid) await h.eventTarget.emit(event);
  assert.equal(conversions, 0);
  assert.strictEqual(h.source.getFrame(), pending);
  h.source.stop();
  await rejected;
});

test('retries stop on timeout and timeout permits a fresh request', async t => {
  t.mock.timers.enable({ apis: ['setInterval', 'setTimeout'] });
  const h = harness(t);
  h.source.start();
  const pending = h.source.getFrame(1300);
  const rejected = assert.rejects(pending, /沒有收到 VDO.Ninja 影格/);
  assert.equal(h.sent.length, 1);
  t.mock.timers.tick(500);
  assert.equal(h.sent.length, 2);
  t.mock.timers.tick(500);
  assert.equal(h.sent.length, 3);
  t.mock.timers.tick(300);
  await rejected;
  assert.equal(h.source.pending, null);
  t.mock.timers.tick(5000);
  assert.equal(h.sent.length, 3);
  const next = h.source.getFrame();
  const cancelled = assert.rejects(next, { name: 'AbortError' });
  assert.equal(h.sent.length, 4);
  h.source.stop();
  await cancelled;
});

test('stop rejects pending frame, removes listener and rejects future capture calls', async t => {
  const h = harness(t);
  await assert.rejects(h.source.getFrame(), { name: 'AbortError' });
  assert.equal(h.sent.length, 0);
  h.source.start();
  const pending = h.source.getFrame();
  const cancelled = assert.rejects(pending, { name: 'AbortError' });
  h.source.stop();
  h.source.stop();
  await cancelled;
  assert.equal(h.source.pending, null);
  assert.equal(h.eventTarget.listeners.size, 0);
  await assert.rejects(h.source.getFrame(), { name: 'AbortError' });
});

test('conversion error rejects only its pending request', async t => {
  const h = harness(t, async () => { throw new Error('decode failed'); });
  h.source.start();
  const pending = h.source.getFrame();
  const failed = assert.rejects(pending, /decode failed/);
  await h.eventTarget.emit(reply(h));
  await failed;
  assert.equal(h.source.pending, null);
});

test('capture diagnostics separate the iframe wait from conversion and clear on stop', async t => {
  let now = 100;
  const h = harness(t, async () => { now += 12; return new Blob(['jpeg']); });
  h.source.now = () => now;
  h.source.start();
  const pending = h.source.getFrame();
  now += 85;
  await h.eventTarget.emit(reply(h));
  await pending;
  assert.deepEqual(h.source.lastTimings, { capture_wait_ms: 85, convert_ms: 12, source_bytes: 3 });
  h.source.stop();
  assert.equal(h.source.lastTimings, null);
});

test('late async conversion after stop/restart cannot resolve a newer request', async t => {
  let resolveOld;
  let conversions = 0;
  const oldJpeg = new Blob(['old'], { type: 'image/jpeg' });
  const newJpeg = new Blob(['new'], { type: 'image/jpeg' });
  const h = harness(t, () => {
    conversions += 1;
    return conversions === 1 ? new Promise(resolve => { resolveOld = resolve; }) : Promise.resolve(newJpeg);
  });
  h.source.start();
  const old = h.source.getFrame();
  const oldRejected = assert.rejects(old, { name: 'AbortError' });
  const oldMessage = reply(h);
  const oldHandler = h.eventTarget.emit(oldMessage);
  // A duplicate response must not start a second simultaneous conversion.
  await h.eventTarget.emit(oldMessage);
  assert.equal(conversions, 1);
  h.source.stop();
  await oldRejected;
  h.source.start();
  const fresh = h.source.getFrame();
  resolveOld(oldJpeg);
  await oldHandler;
  assert.strictEqual(h.source.getFrame(), fresh);
  await h.eventTarget.emit(oldMessage);
  assert.equal(conversions, 1);
  await h.eventTarget.emit(reply(h));
  assert.strictEqual(await fresh, newJpeg);
  assert.equal(conversions, 2);
});

test('late conversion rejection after timeout does not reject a replacement request', async t => {
  t.mock.timers.enable({ apis: ['setInterval', 'setTimeout'] });
  let rejectOld;
  let conversions = 0;
  const h = harness(t, () => {
    conversions += 1;
    return conversions === 1 ? new Promise((resolve, reject) => { rejectOld = reject; }) : Promise.resolve(new Blob(['new']));
  });
  h.source.start();
  const old = h.source.getFrame(10);
  const timedOut = assert.rejects(old, /沒有收到/);
  const oldHandler = h.eventTarget.emit(reply(h));
  t.mock.timers.tick(10);
  await timedOut;
  const fresh = h.source.getFrame();
  rejectOld(new Error('late decode failure'));
  await oldHandler;
  assert.strictEqual(h.source.getFrame(), fresh);
  await h.eventTarget.emit(reply(h));
  assert.equal((await fresh).size, 3);
});

test('synchronous iframe postMessage failure rejects promise and clears its timers', async t => {
  t.mock.timers.enable({ apis: ['setInterval', 'setTimeout'] });
  const h = harness(t);
  let attempts = 0;
  h.contentWindow.postMessage = () => { attempts += 1; throw new Error('iframe was destroyed'); };
  h.source.start();
  let pending;
  assert.doesNotThrow(() => { pending = h.source.getFrame(); });
  await assert.rejects(pending, /iframe was destroyed/);
  assert.equal(h.source.pending, null);
  t.mock.timers.tick(20_000);
  assert.equal(attempts, 1);
});
