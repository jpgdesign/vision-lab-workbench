import test from 'node:test';
import assert from 'node:assert/strict';
import { VdoPeerSource } from '../src/vdoPeerSource.js';

class Track extends EventTarget {
  constructor(kind = 'video') { super(); this.kind = kind; this.readyState = 'live'; this.stops = 0; }
  stop() { this.stops++; this.readyState = 'ended'; }
}
class Stream {
  tracks = [];
  addTrack(track) { this.tracks.push(track); }
  removeTrack(track) { this.tracks = this.tracks.filter(item => item !== track); }
  getTracks() { return [...this.tracks]; }
  getVideoTracks() { return this.tracks.filter(track => track.kind === 'video'); }
}
class SDK extends EventTarget {
  constructor(options) { super(); this.options = options; this.viewCalls = []; this.disconnects = 0; this.stats = {}; }
  async connect() { await this.connectWait; if (this.connectError) throw this.connectError; }
  async view(...args) { this.viewCalls.push(args); return {}; }
  async disconnect() { this.disconnects++; }
  async getStats() { return this.stats; }
  event(name, detail) { this.dispatchEvent(new CustomEvent(name, { detail })); }
}
function fixture(options = {}) {
  const events = [];
  const errors = [];
  const source = new VdoPeerSource({
    source: { url: 'https://vdo.ninja/?view=test_camera' },
    SDKClass: SDK, MediaStreamClass: Stream,
    onStream: stream => events.push(stream), onError: error => errors.push(error),
    ...options,
  });
  return { source, events, errors };
}
const tick = () => new Promise(resolve => setImmediate(resolve));

test('browser SDK export loads without Node WebRTC dependencies', () => {
  const source = new VdoPeerSource({ source: 'https://vdo.ninja/?view=test_camera', MediaStreamClass: Stream });
  assert.equal(typeof source.SDKClass, 'function');
  assert.equal(source.SDKClass.VERSION, '1.6.2');
});

test('direct viewer uses official salt and requests only video, without file/resource permissions', async () => {
  const { source, events } = fixture();
  const pending = source.start();
  await tick();
  assert.equal(source.sdk.options.salt, 'vdo.ninja');
  assert.equal(source.sdk.options.debug, false);
  assert.equal(Object.hasOwn(source.sdk.options, 'password'), false);
  assert.deepEqual(source.sdk.viewCalls, [['test_camera', { audio: false, video: true, downloads: false, allowresources: false }]]);
  let settled = false;
  pending.then(() => { settled = true; });
  await tick();
  assert.equal(settled, false, 'view() completion must not indicate that a video track exists');
  const audio = new Track('audio');
  source.sdk.event('track', { track: audio });
  assert.equal(audio.stops, 1);
  assert.equal(events.length, 0);
  const video = new Track();
  source.sdk.event('track', { track: video, streams: [] });
  assert.equal(await pending, source.stream);
  assert.deepEqual(source.stream.getVideoTracks(), [video]);
  assert.equal(events.length, 1);
  await source.stop();
  assert.equal(video.stops, 1);
});

test('password aliases preserve VDO double-decode and explicit false semantics', async () => {
  for (const key of ['password', 'pass', 'pw', 'p']) {
    const { source } = fixture({ source: `https://vdo.ninja/?view=test_camera&${key}=literal%2520pass&salt=custom` });
    const pending = source.start();
    assert.equal(source.sdk.options.password, 'literal pass');
    assert.equal(source.sdk.options.salt, 'custom');
    const rejection = assert.rejects(pending, { name: 'AbortError' });
    await source.stop();
    await rejection;
  }
  for (const value of ['false', '0', 'off']) {
    const { source } = fixture({ source: `https://vdo.ninja/?view=test_camera&password=${value}` });
    const pending = source.start();
    assert.equal(source.sdk.options.password, false);
    const rejection = assert.rejects(pending, { name: 'AbortError' });
    await source.stop();
    await rejection;
  }
});

test('rejects blank prompted passwords and SDK stream identity changes before connecting', () => {
  assert.throws(() => fixture({ source: 'https://vdo.ninja/?view=test_camera&password=' }), /完整密碼/);
  assert.throws(() => fixture({ source: `https://vdo.ninja/?view=${'a'.repeat(65)}` }), /串流 ID/);
  assert.throws(() => fixture({ source: 'https://vdo.ninja/?view=other-camera' }), /串流 ID/);
  assert.throws(() => fixture({ source: 'https://evil.example/?view=test_camera' }), /VIEW/);
});

test('stop during connection rejects startup, suppresses late view and removes listeners', async () => {
  let release;
  class DeferredSDK extends SDK { connectWait = new Promise(resolve => { release = resolve; }); }
  const { source, errors, events } = fixture({ SDKClass: DeferredSDK });
  const pending = source.start();
  const rejection = assert.rejects(pending, { name: 'AbortError' });
  const stopped = source.stop();
  assert.equal(source.stop(), stopped);
  await stopped;
  release();
  await rejection;
  await tick();
  source.sdk.event('track', { track: new Track() });
  assert.equal(source.sdk.viewCalls.length, 0);
  assert.equal(source.sdk.disconnects, 1);
  assert.equal(source.listeners.size, 0);
  assert.equal(events.length, 0);
  assert.equal(errors.length, 0);
  await assert.rejects(source.start(), { name: 'AbortError' });
});

test('startup timeout disconnects and reports a safe actionable error', async () => {
  const { source, errors } = fixture({ timeoutMs: 10 });
  await assert.rejects(source.start(), /沒有收到 VDO.Ninja/);
  await source.stop();
  assert.equal(source.sdk.disconnects, 1);
  assert.equal(errors.length, 0);
});

test('startup SDK failure never exposes private URL or password in error', async () => {
  class BrokenSDK extends SDK { connectError = new Error('wss://private/?password=SECRET'); }
  const { source } = fixture({ SDKClass: BrokenSDK });
  await assert.rejects(source.start(), error => !error.message.includes('SECRET') && error.message.includes('連線失敗'));
  await source.stop();
});

test('replacement tracks stop old video and ignore duplicate or foreign stream tracks', async () => {
  const { source, events } = fixture();
  const pending = source.start();
  const first = new Track();
  source.sdk.event('track', { track: first, streamID: 'test_camera' });
  await pending;
  source.sdk.event('track', { track: first, streamID: 'test_camera' });
  const foreign = new Track();
  source.sdk.event('track', { track: foreign, streamID: 'different_camera' });
  assert.equal(foreign.stops, 1);
  const second = new Track();
  source.sdk.event('track', { track: second, streamID: 'test_camera' });
  assert.equal(first.stops, 1);
  assert.deepEqual(source.stream.getVideoTracks(), [second]);
  assert.equal(events.length, 2);
  await source.stop();
});

test('temporary signaling loss preserves media; sustained peer failure reports once and cleans up', async () => {
  const { source, errors } = fixture({ disconnectGraceMs: 15 });
  const pending = source.start();
  source.sdk.event('track', { track: new Track() });
  await pending;
  source.sdk.event('disconnected', { intentional: false, willReconnect: true });
  assert.equal(source.disconnectTimer, null);
  source.sdk.event('connectionFailed', {});
  source.sdk.event('connectionRecovered', {});
  assert.equal(source.disconnectTimer, null);
  source.sdk.event('connectionFailed', {});
  await new Promise(resolve => setTimeout(resolve, 25));
  assert.equal(errors.length, 1);
  assert.equal(source.active, false);
  await source.stop();
  assert.equal(source.sdk.disconnects, 1);
});

test('stats isolate received video and derive real FPS/bitrate only from consecutive samples', async () => {
  const { source } = fixture();
  const pending = source.start();
  source.sdk.event('track', { track: new Track() });
  await pending;
  const video = { id: 'video', type: 'inbound-rtp', connectionType: 'viewer', kind: 'video', frameWidth: 1280, frameHeight: 720, framesDecoded: 100, timestamp: 1000, bytesReceived: 10000 };
  source.sdk.stats = { peer: [video, { ...video, connectionType: 'publisher', framesPerSecond: 999, bytesReceived: 999999 }] };
  assert.deepEqual(await source.getStats(), { fps: null, width: 1280, height: 720, bitrateKbps: null });
  Object.assign(video, { framesDecoded: 160, framesPerSecond: 60, timestamp: 3000, bytesReceived: 210000 });
  assert.deepEqual(await source.getStats(), { fps: 30, width: 1280, height: 720, bitrateKbps: 800 });
  Object.assign(video, { framesPerSecond: 30, timestamp: 4000, bytesReceived: 210000 });
  assert.equal((await source.getStats()).fps, 0, 'a stalled decoded-frame counter overrides stale reported FPS');
  delete video.framesPerSecond;
  Object.assign(video, { framesDecoded: 1, bytesReceived: 100, timestamp: 5000 });
  assert.equal((await source.getStats()).fps, null, 'counter reset is not a negative FPS');
  await source.stop();
  assert.deepEqual(await source.getStats(), { fps: null, width: null, height: null, bitrateKbps: null });
});

test('stats use reported FPS only before a usable decoded-frame interval exists', async () => {
  const { source } = fixture();
  const pending = source.start();
  source.sdk.event('track', { track: new Track() });
  await pending;
  const video = { id: 'video', type: 'inbound-rtp', connectionType: 'viewer', kind: 'video', framesPerSecond: 24, framesDecoded: 100, timestamp: 1000 };
  source.sdk.stats = { peer: [video] };
  assert.equal((await source.getStats()).fps, 24, 'first sample has no counter interval');
  Object.assign(video, { framesDecoded: 120, timestamp: 2000 });
  assert.equal((await source.getStats()).fps, 20, 'valid decoded interval overrides reported FPS');
  Object.assign(video, { framesDecoded: 1, timestamp: 3000, framesPerSecond: 15 });
  assert.equal((await source.getStats()).fps, 15, 'a reset counter falls back to browser FPS');
  delete video.framesDecoded;
  Object.assign(video, { timestamp: 4000, framesPerSecond: 0 });
  assert.equal((await source.getStats()).fps, 0, 'missing counters preserve a valid browser zero');
  await source.stop();
});

test('a stats response arriving after stop cannot revive measurements', async () => {
  const { source } = fixture();
  const pending = source.start();
  source.sdk.event('track', { track: new Track() });
  await pending;
  let release;
  source.sdk.getStats = () => new Promise(resolve => { release = resolve; });
  const stats = source.getStats();
  await source.stop();
  release({ peer: [{ id: 'v', type: 'inbound-rtp', connectionType: 'viewer', kind: 'video', framesPerSecond: 30 }] });
  assert.equal((await stats).fps, null);
});
