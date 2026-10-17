import test from 'node:test';
import assert from 'node:assert/strict';
import { LiveJpegEncoder } from '../src/liveJpegEncoder.js';

const jpeg = () => new Blob(['jpeg bytes'], { type: 'image/jpeg' });
const video = (width = 1920, height = 1080) => ({ videoWidth: width, videoHeight: height, readyState: 2 });
const deferred = () => { let resolve, reject; const promise = new Promise((ok, fail) => { resolve = ok; reject = fail; }); return { promise, resolve, reject }; };
function surfaces({ convert = () => Promise.resolve(jpeg()), toBlob = callback => callback(jpeg()) } = {}) {
  const instances = [];
  class Surface {
    constructor(width = 1, height = 1) {
      this.w = width; this.h = height; this.resizes = []; this.draws = []; this.encodes = []; this.contextCalls = 0;
      this.context = { drawImage: (...args) => this.draws.push(args) }; instances.push(this);
    }
    get width() { return this.w; } set width(value) { this.w = value; this.resizes.push(['width', value]); }
    get height() { return this.h; } set height(value) { this.h = value; this.resizes.push(['height', value]); }
    getContext(type, options) { this.contextCalls++; assert.equal(type, '2d'); assert.equal(options.alpha, false); return this.context; }
    convertToBlob(options) { assert.ok(this instanceof Surface, 'native-style encoder receiver preserved'); this.encodes.push(options); return convert(); }
    toBlob(callback, type, quality) { assert.ok(this instanceof Surface, 'native-style encoder receiver preserved'); this.encodes.push({ type, quality }); toBlob(callback); }
  }
  return { Surface, instances };
}

test('OffscreenCanvas scales within 960 × 720, encodes JPEG, and reuses its surface/context without per-frame resize', async () => {
  const { Surface, instances } = surfaces(); let fallbacks = 0;
  const encoder = new LiveJpegEncoder({ OffscreenCanvasClass: Surface, getFallbackCanvas: () => { fallbacks++; } });
  await encoder.encode(video()); await encoder.encode(video());
  assert.equal(instances.length, 1); assert.equal(fallbacks, 0);
  const canvas = instances[0];
  assert.deepEqual(canvas.resizes, [['width', 960], ['height', 540]]);
  assert.equal(canvas.contextCalls, 1); assert.equal(canvas.draws.length, 2);
  assert.deepEqual(canvas.encodes, [{ type: 'image/jpeg', quality: 0.76 }, { type: 'image/jpeg', quality: 0.76 }]);
  await encoder.encode(video(1080, 1920));
  assert.equal(canvas.width, 405); assert.equal(canvas.height, 720);
  await encoder.encode(video(320, 240));
  assert.equal(canvas.width, 320); assert.equal(canvas.height, 240);
});

test('HTML fallback is reused when OffscreenCanvas is absent and does not upscale small frames', async () => {
  const { Surface } = surfaces(); const html = new Surface(); let gets = 0;
  const encoder = new LiveJpegEncoder({ OffscreenCanvasClass: null, getFallbackCanvas: () => { gets++; return html; } });
  await encoder.encode(video(640, 480)); await encoder.encode(video(640, 480));
  assert.equal(gets, 1); assert.equal(html.contextCalls, 1);
  assert.deepEqual(html.resizes, [['width', 640], ['height', 480]]);
  assert.deepEqual(html.encodes[0], { type: 'image/jpeg', quality: 0.76 });
});

test('unsupported OffscreenCanvas conversion falls back once and remains on HTML canvas', async () => {
  const offscreen = surfaces({ convert: () => Promise.reject(new DOMException('unsupported', 'NotSupportedError')) });
  const html = new (surfaces().Surface)();
  const encoder = new LiveJpegEncoder({ OffscreenCanvasClass: offscreen.Surface, getFallbackCanvas: () => html });
  await encoder.encode(video()); await encoder.encode(video());
  assert.equal(offscreen.instances.length, 1); assert.equal(offscreen.instances[0].encodes.length, 1);
  assert.equal(html.encodes.length, 2); assert.equal(html.contextCalls, 1);
});

test('an unavailable OffscreenCanvas constructor falls back without failing the session', async () => {
  const html = new (surfaces().Surface)();
  class Unavailable { constructor() { throw new Error('unavailable'); } }
  const encoder = new LiveJpegEncoder({ OffscreenCanvasClass: Unavailable, getFallbackCanvas: () => html });
  assert.equal((await encoder.encode(video())).type, 'image/jpeg');
});

test('one encode may run at a time, with no pending-frame queue', async () => {
  const pending = deferred(), { Surface, instances } = surfaces({ convert: () => pending.promise });
  const encoder = new LiveJpegEncoder({ OffscreenCanvasClass: Surface });
  const first = encoder.encode(video());
  await assert.rejects(encoder.encode(video()), { name: 'InvalidStateError' });
  assert.equal(instances[0].draws.length, 1);
  pending.resolve(jpeg()); await first;
  await encoder.encode(video()); assert.equal(instances[0].draws.length, 2);
});

test('dispose ignores late successful encodes and prevents new capture work', async () => {
  const pending = deferred(), { Surface, instances } = surfaces({ convert: () => pending.promise });
  const encoder = new LiveJpegEncoder({ OffscreenCanvasClass: Surface });
  const first = encoder.encode(video()); encoder.dispose(); pending.resolve(jpeg());
  await assert.rejects(first, { name: 'AbortError' });
  await assert.rejects(encoder.encode(video()), { name: 'AbortError' });
  assert.equal(instances[0].draws.length, 1);
});

test('dispose during failed OffscreenCanvas conversion never starts a fallback encode', async () => {
  const pending = deferred(), { Surface } = surfaces({ convert: () => pending.promise }); let fallbacks = 0;
  const encoder = new LiveJpegEncoder({ OffscreenCanvasClass: Surface, getFallbackCanvas: () => { fallbacks++; } });
  const first = encoder.encode(video()); encoder.dispose(); pending.reject(new Error('late failure'));
  await assert.rejects(first, { name: 'AbortError' }); assert.equal(fallbacks, 0);
});

test('late HTML callback after disposal cannot provide a usable frame', async () => {
  let callback; const html = new (surfaces({ toBlob: value => { callback = value; } }).Surface)();
  const encoder = new LiveJpegEncoder({ OffscreenCanvasClass: null, getFallbackCanvas: () => html });
  const first = encoder.encode(video()); encoder.dispose(); callback(jpeg());
  await assert.rejects(first, { name: 'AbortError' });
});

test('invalid video and JPEG outputs fail before upload', async () => {
  for (const output of [null, new Blob([], { type: 'image/jpeg' }), new Blob(['png'], { type: 'image/png' }), new Blob([new Uint8Array(1024 * 1024 + 1)], { type: 'image/jpeg' })]) {
    const { Surface } = surfaces({ convert: () => Promise.resolve(output) });
    await assert.rejects(new LiveJpegEncoder({ OffscreenCanvasClass: Surface }).encode(video()), /1 MB/);
  }
  const { Surface, instances } = surfaces();
  const encoder = new LiveJpegEncoder({ OffscreenCanvasClass: Surface });
  await assert.rejects(encoder.encode(video(0, 0)), /影格/);
  await assert.rejects(encoder.encode({ ...video(), readyState: 1 }), /影格/);
  assert.equal(instances.length, 0);
});

test('security errors are not retried through another canvas', async () => {
  const { Surface } = surfaces({ convert: () => Promise.reject(new DOMException('tainted', 'SecurityError')) }); let fallbacks = 0;
  const encoder = new LiveJpegEncoder({ OffscreenCanvasClass: Surface, getFallbackCanvas: () => { fallbacks++; } });
  await assert.rejects(encoder.encode(video()), { name: 'SecurityError' }); assert.equal(fallbacks, 0);
});
