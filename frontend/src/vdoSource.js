// Official VDO.Ninja iframe API: a viewer frame replies with a structured-cloned
// PNG Blob. No cross-origin DOM access, proxy, publisher, microphone or camera.
const VDO_ORIGIN = 'https://vdo.ninja';
let requestNumber = 0;

export function parseVdoViewUrl(value, parentOrigin = globalThis.location?.origin || 'http://127.0.0.1:8765') {
  let input;
  try { input = new URL(String(value).trim()); }
  catch { throw new Error('請貼上完整的 https://vdo.ninja/?view=… 觀看連結。'); }
  if (input.origin !== VDO_ORIGIN || !['', '/'].includes(input.pathname) || input.username || input.password || input.hash) {
    throw new Error('目前僅接受 https://vdo.ninja/ 的 VIEW 觀看連結。');
  }
  const ids = input.searchParams.getAll('view');
  if (ids.length !== 1 || !/^[a-zA-Z0-9_-]{1,128}$/.test(ids[0])) {
    throw new Error('連結必須包含一個有效的 view 串流 ID，請使用手機提供的觀看連結。');
  }
  if (['push', 'publish', 'director', 'room'].some(key => input.searchParams.has(key))) {
    throw new Error('請使用單一畫面的 VIEW 觀看連結，不是發布或導播連結。');
  }
  const url = new URL('/', VDO_ORIGIN);
  url.searchParams.set('view', ids[0]);
  // Retain only documented password aliases and stream salt, never arbitrary
  // viewer commands, remote URLs, scripts or publishing options.
  for (const key of ['password', 'pass', 'pw', 'p', 'salt']) {
    if (input.searchParams.has(key)) {
      const item = input.searchParams.get(key);
      if (item.length > 256 || input.searchParams.getAll(key).length !== 1) throw new Error('觀看連結的密碼參數格式不正確。');
      url.searchParams.set(key, item);
    }
  }
  url.searchParams.set('autostart', '');
  url.searchParams.set('cleanoutput', '');
  url.searchParams.set('noaudio', '');
  url.searchParams.set('muted', '');
  url.searchParams.set('iframetarget', parentOrigin);
  return { url: url.href, streamId: ids[0], origin: VDO_ORIGIN };
}

export async function frameToJpeg(blob) {
  if (!(blob instanceof Blob) || !['image/png', 'image/jpeg'].includes(blob.type) || !blob.size || blob.size > 20 * 1024 * 1024) {
    throw new Error('VDO.Ninja 影格無效或過大，請降低手機串流解析度。');
  }
  const bitmap = await createImageBitmap(blob);
  try {
    if (!bitmap.width || !bitmap.height || bitmap.width * bitmap.height > 33_177_600) throw new Error('VDO.Ninja 影格尺寸超出支援範圍。');
    const scale = Math.min(960 / bitmap.width, 720 / bitmap.height, 1);
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.floor(bitmap.width * scale));
    canvas.height = Math.max(1, Math.floor(bitmap.height * scale));
    canvas.getContext('2d', { alpha: false }).drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    const jpeg = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.76));
    if (!jpeg || jpeg.size > 1024 * 1024) throw new Error('VDO.Ninja 影格無法轉為 1 MB 以下的 JPEG。');
    return jpeg;
  } finally { bitmap.close(); }
}

export class VdoFrameSource {
  constructor({ iframe, source, eventTarget = globalThis.window, convert = frameToJpeg, now = () => performance.now() }) {
    this.iframe = iframe;
    this.source = source;
    this.eventTarget = eventTarget;
    this.convert = convert;
    this.now = now;
    this.lastTimings = null;
    this.active = false;
    this.pending = null;
    this.onMessage = this.onMessage.bind(this);
  }
  start() {
    if (!this.active) { this.active = true; this.eventTarget.addEventListener('message', this.onMessage); }
  }
  getFrame(timeoutMs = 12000) {
    if (!this.active) return Promise.reject(new DOMException('VDO.Ninja 來源已停止。', 'AbortError'));
    if (this.pending) return this.pending.promise;
    const cib = `vision-${Date.now()}-${++requestNumber}`;
    let resolve, reject;
    const promise = new Promise((ok, fail) => { resolve = ok; reject = fail; });
    const pending = { cib, promise, resolve, reject, converting: false, startedAt: this.now() };
    this.pending = pending;
    const send = () => {
      if (this.active && this.pending === pending && !pending.converting) {
        try {
          this.iframe.contentWindow?.postMessage({ getVideoFrame: true, streamID: this.source.streamId, cib }, this.source.origin);
        } catch (error) { this.finish(pending, error); }
      }
    };
    pending.retry = setInterval(send, 500);
    pending.timeout = setTimeout(() => this.finish(pending, new Error('沒有收到 VDO.Ninja 影格。請確認手機仍在分享、VIEW 連結正確，並保持手機頁面開啟。')), timeoutMs);
    send();
    return promise;
  }
  async onMessage(event) {
    const pending = this.pending;
    if (!this.active || !pending || pending.converting || event.origin !== this.source.origin || event.source !== this.iframe.contentWindow) return;
    const data = event.data;
    if (!data || data.action !== 'image-frame-capture') return;
    const value = data.value;
    if (!value || value.cib !== pending.cib || value.streamID !== this.source.streamId) return;
    // A stream that is still connecting can respond with a null/empty Blob.
    if (!(value.imageData instanceof Blob) || !value.imageData.size) return;
    pending.converting = true;
    const receivedAt = this.now();
    try {
      const jpeg = await this.convert(value.imageData);
      if (this.active && this.pending === pending) {
        this.lastTimings = { capture_wait_ms: receivedAt - pending.startedAt,
          convert_ms: this.now() - receivedAt, source_bytes: value.imageData.size };
        this.finish(pending, null, jpeg);
      }
    } catch (error) {
      if (this.pending === pending) this.finish(pending, error);
    }
  }
  finish(pending, error, value) {
    if (this.pending !== pending) return;
    clearTimeout(pending.timeout);
    clearInterval(pending.retry);
    this.pending = null;
    if (error) pending.reject(error); else pending.resolve(value);
  }
  stop() {
    this.active = false;
    this.lastTimings = null;
    this.eventTarget.removeEventListener('message', this.onMessage);
    if (this.pending) this.finish(this.pending, new DOMException('VDO.Ninja 來源已停止。', 'AbortError'));
  }
}
