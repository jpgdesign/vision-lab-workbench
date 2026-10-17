import VDONinjaSDK from '@vdoninja/sdk/browser';
import { parseVdoViewUrl } from './vdoSource.js';

const emptyStats = () => ({ fps: null, width: null, height: null, bitrateKbps: null });
const finite = value => typeof value === 'number' && Number.isFinite(value) && value >= 0;
const aborted = () => new DOMException('VDO.Ninja 來源已停止。', 'AbortError');

function connectionOptions(source) {
  const parsed = parseVdoViewUrl(typeof source === 'string' ? source : source?.url);
  // The SDK normalizes punctuation and truncates IDs; reject changes to the
  // requested identity instead of silently connecting to a different stream.
  if (!/^[a-zA-Z0-9_]{1,64}$/.test(parsed.streamId)) {
    throw new Error('直接串流目前支援 1–64 字元的英數或底線串流 ID；此連結請使用相容影格模式。');
  }
  const query = new URL(parsed.url).searchParams;
  const aliases = ['password', 'pass', 'pw', 'p'];
  let password = aliases.map(key => query.get(key)).find(Boolean);
  if (!password && aliases.some(key => query.has(key))) {
    throw new Error('觀看連結需要密碼，請貼上包含完整密碼的 VIEW 連結。');
  }
  if (['false', '0', 'off'].includes(password)) password = false;
  else if (password !== undefined) {
    // VDO.Ninja performs one additional decoding pass after URLSearchParams.
    try { password = decodeURIComponent(password); } catch { /* Keep literal %. */ }
  }
  return {
    streamId: parsed.streamId,
    options: {
      salt: query.get('salt') || 'vdo.ninja',
      ...(password === undefined ? {} : { password }),
      debug: false,
      allowresources: false,
      allowchunked: false,
    },
  };
}

// Receive the official SDK's decoded video track directly. This removes the
// iframe PNG encode/decode round trip and uses the ordinary camera JPEG path.
export class VdoPeerSource {
  constructor({ source, onStream = () => {}, onError = () => {}, SDKClass = VDONinjaSDK,
    MediaStreamClass = globalThis.MediaStream, timeoutMs = 30000, disconnectGraceMs = 8000 }) {
    this.config = connectionOptions(source);
    this.SDKClass = SDKClass;
    this.MediaStreamClass = MediaStreamClass;
    this.onStream = onStream;
    this.onError = onError;
    this.timeoutMs = timeoutMs;
    this.disconnectGraceMs = disconnectGraceMs;
    this.stream = null;
    this.active = false;
    this.ready = false;
    this.sdk = null;
    this.listeners = new Map();
    this.trackListeners = new Map();
    this.samples = new Map();
    this.startPromise = null;
    this.stopPromise = null;
  }

  start() {
    if (this.stopPromise) return Promise.reject(aborted());
    if (this.startPromise) return this.startPromise;
    this.active = true;
    this.startPromise = new Promise((resolve, reject) => {
      this.resolveStart = resolve;
      this.rejectStart = reject;
    });
    try {
      this.stream = new this.MediaStreamClass();
      this.sdk = new this.SDKClass(this.config.options);
      this.listen('track', event => this.receiveTrack(event.detail || {}));
      this.listen('connectionRecovered', () => this.clearDisconnectTimer());
      this.listen('connectionFailed', () => this.scheduleFailure());
      this.listen('reconnectFailed', () => this.scheduleFailure());
      this.listen('disconnected', event => {
        // A signaling socket reconnect need not interrupt established media.
        if (!event.detail?.intentional && event.detail?.willReconnect === false) this.scheduleFailure();
      });
      this.timeout = setTimeout(() => this.fail(new Error('30 秒內沒有收到 VDO.Ninja 影片。請確認手機持續分享、觀看連結與密碼正確；也可改用相容影格模式。')), this.timeoutMs);
      this.connect();
    } catch {
      this.fail(new Error('此瀏覽器無法建立 VDO.Ninja 直接串流，請使用 Chrome／Edge 或相容影格模式。'));
    }
    return this.startPromise;
  }

  async connect() {
    try {
      await this.sdk.connect();
      if (!this.active) return;
      await this.sdk.view(this.config.streamId, {
        audio: false, video: true, downloads: false, allowresources: false,
      });
      // view() can resolve before tracks arrive, or return null when offline.
      // The track event and overall timeout decide whether the source is ready.
    } catch {
      if (this.active) this.fail(new Error('VDO.Ninja 直接串流連線失敗，請檢查手機與網路，或使用相容影格模式。'));
    }
  }

  listen(name, handler) {
    this.listeners.set(name, handler);
    this.sdk.addEventListener(name, handler);
  }

  receiveTrack({ track, streamID }) {
    if (!track) return;
    if (!this.active || track.kind !== 'video' || track.readyState === 'ended' ||
      (streamID && streamID !== this.config.streamId)) {
      track.stop();
      return;
    }
    if (this.stream.getVideoTracks().includes(track)) return;
    this.clearDisconnectTimer();
    for (const previous of this.stream.getVideoTracks()) {
      previous.removeEventListener('ended', this.trackListeners.get(previous));
      this.trackListeners.delete(previous);
      this.stream.removeTrack(previous);
      previous.stop();
    }
    this.stream.addTrack(track);
    const ended = () => { if (this.active) this.scheduleFailure(); };
    this.trackListeners.set(track, ended);
    track.addEventListener('ended', ended);
    try { this.onStream(this.stream); }
    catch {
      this.fail(new Error('無法顯示 VDO.Ninja 直接串流，請重新連線。'));
      return;
    }
    if (!this.ready) {
      this.ready = true;
      clearTimeout(this.timeout);
      this.resolveStart?.(this.stream);
      this.resolveStart = null;
      this.rejectStart = null;
    }
  }

  clearDisconnectTimer() {
    clearTimeout(this.disconnectTimer);
    this.disconnectTimer = null;
  }

  scheduleFailure() {
    if (!this.active) return;
    if (!this.ready) {
      this.fail(new Error('VDO.Ninja 直接串流無法連線，請確認手機分享狀態，或改用相容影格模式。'));
      return;
    }
    if (!this.disconnectTimer) this.disconnectTimer = setTimeout(() => {
      this.fail(new Error('VDO.Ninja 影片連線已中斷，請確認手機與網路後重新連線。'));
    }, this.disconnectGraceMs);
  }

  fail(error) {
    if (!this.active) return;
    const notify = this.ready;
    this.rejectStart?.(error);
    this.resolveStart = null;
    this.rejectStart = null;
    void this.stop();
    if (notify) { try { this.onError(error); } catch { /* Already stopped. */ } }
  }

  async getStats() {
    if (!this.active || !this.ready || !this.sdk) return emptyStats();
    let reports;
    try { reports = await this.sdk.getStats(); } catch { return emptyStats(); }
    if (!this.active) return emptyStats();
    let selected = null;
    for (const [peerId, rows] of Object.entries(reports || {})) {
      if (!Array.isArray(rows)) continue;
      for (const row of rows) {
        if (!row || row.connectionType !== 'viewer' || row.type !== 'inbound-rtp' ||
          (row.kind || row.mediaType) !== 'video') continue;
        if (!selected || (row.bytesReceived || 0) > (selected.row.bytesReceived || 0)) selected = { peerId, row };
      }
    }
    if (!selected) return emptyStats();
    const { peerId, row } = selected;
    const key = `${peerId}:${row.id}`;
    const previous = this.samples.get(key);
    const deltaMs = previous && finite(row.timestamp) ? row.timestamp - previous.timestamp : 0;
    let fps = finite(row.framesPerSecond) ? row.framesPerSecond : null;
    let bitrateKbps = null;
    if (deltaMs > 0 && deltaMs <= 10000) {
      // The browser can retain framesPerSecond while playback has stalled.
      // Prefer actual decoded-frame progress across this sampling interval.
      if (finite(row.framesDecoded) && finite(previous.framesDecoded) && row.framesDecoded >= previous.framesDecoded) {
        fps = (row.framesDecoded - previous.framesDecoded) * 1000 / deltaMs;
      }
      if (finite(row.bytesReceived) && finite(previous.bytesReceived) && row.bytesReceived >= previous.bytesReceived) {
        bitrateKbps = (row.bytesReceived - previous.bytesReceived) * 8 / deltaMs;
      }
    }
    this.samples.clear();
    this.samples.set(key, { timestamp: row.timestamp, framesDecoded: row.framesDecoded, bytesReceived: row.bytesReceived });
    return {
      fps,
      width: finite(row.frameWidth) && row.frameWidth > 0 ? row.frameWidth : null,
      height: finite(row.frameHeight) && row.frameHeight > 0 ? row.frameHeight : null,
      bitrateKbps,
    };
  }

  stop() {
    if (this.stopPromise) return this.stopPromise;
    this.active = false;
    clearTimeout(this.timeout);
    this.clearDisconnectTimer();
    this.rejectStart?.(aborted());
    this.resolveStart = null;
    this.rejectStart = null;
    for (const [name, handler] of this.listeners) this.sdk?.removeEventListener(name, handler);
    this.listeners.clear();
    for (const [track, handler] of this.trackListeners) track.removeEventListener('ended', handler);
    this.trackListeners.clear();
    for (const track of this.stream?.getTracks() || []) track.stop();
    this.samples.clear();
    this.stopPromise = Promise.resolve().then(() => this.sdk?.disconnect()).catch(() => {});
    return this.stopPromise;
  }
}
