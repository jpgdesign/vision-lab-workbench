// One capture and one upload may overlap. Between them, only the newest
// completed frame is retained; a slow upload never creates an image queue.
export class LatestFramePipeline {
  constructor({ fps, capture, upload, onAccepted = () => {}, onError = () => {}, timeoutMs = 10000,
    now = () => performance.now(), setTimer = (callback, delay) => setTimeout(callback, delay), clearTimer = id => clearTimeout(id),
    makeController = () => new AbortController() }) {
    if (!Number.isFinite(fps) || fps < 1 || fps > 30) throw new RangeError('FPS 必須介於 1 與 30。');
    this.period = 1000 / fps;
    Object.assign(this, { capture, upload, onAccepted, onError, timeoutMs, now, setTimer, clearTimer, makeController });
    this.active = false; this.started = false; this.epoch = 0;
    this.capturing = false; this.uploading = false; this.pending = null; this.dropped = 0;
    this.captureTimer = null; this.uploadTimer = null; this.uploadTimeout = null; this.controller = null; this.cancelUpload = null;
  }
  start() {
    if (this.active) return;
    if (this.started) throw new Error('停止的影格流程不可重新使用，請建立新的流程。');
    this.started = true; this.active = true; this.epoch += 1;
    this.captureOrigin = this.now(); this.captureSlot = 0; this.uploadOrigin = null; this.uploadSlot = 0; this.backoffUntil = 0; this.discardCapturedBefore = -Infinity;
    this.captureTick(this.captureOrigin, this.epoch);
  }
  stop() {
    this.active = false; this.epoch += 1;
    this.clearTimer(this.captureTimer); this.clearTimer(this.uploadTimer); this.clearTimer(this.uploadTimeout);
    this.captureTimer = null; this.uploadTimer = null; this.uploadTimeout = null;
    this.pending = null;
    this.controller?.abort(); this.cancelUpload?.(); this.controller = null; this.cancelUpload = null;
  }
  isCurrent(epoch) { return this.active && this.epoch === epoch; }
  getState() { return { active: this.active, capturing: this.capturing, uploading: this.uploading, hasPending: Boolean(this.pending), dropped: this.dropped }; }
  captureTick(scheduledAt, epoch) {
    this.captureTimer = null;
    if (!this.isCurrent(epoch)) return;
    const startedAt = this.now();
    if (!this.capturing && startedAt >= this.backoffUntil) {
      this.capturing = true;
      Promise.resolve().then(() => this.isCurrent(epoch) ? this.capture({ scheduledAt, startedAt }) : undefined).then(value => {
        if (!this.isCurrent(epoch)) return;
        const completedAt = this.now();
        if (completedAt < this.backoffUntil || startedAt < this.discardCapturedBefore) { this.dropped += 1; return; }
        if (this.pending) this.dropped += 1;
        this.pending = { value, captureStartedAt: startedAt, captureCompletedAt: completedAt };
        this.dispatch(epoch);
      }).catch(error => { if (this.isCurrent(epoch)) this.fail(error); }).finally(() => { this.capturing = false; });
    }
    // Absolute deadlines avoid accumulating a rounded 33 ms interval at 30
    // FPS. Missed slots are skipped, including after a background-tab pause.
    this.captureSlot = Math.max(this.captureSlot + 1, Math.floor((startedAt - this.captureOrigin) / this.period) + 1,
      Math.ceil((this.backoffUntil - this.captureOrigin) / this.period));
    const nextAt = this.captureOrigin + this.captureSlot * this.period;
    this.captureTimer = this.setTimer(() => this.captureTick(nextAt, epoch), Math.max(0, Math.ceil(nextAt - this.now())));
  }
  dispatch(epoch) {
    if (!this.isCurrent(epoch) || this.uploading || !this.pending) return;
    this.clearTimer(this.uploadTimer); this.uploadTimer = null;
    const now = this.now();
    const deadline = Math.max(this.backoffUntil, this.uploadOrigin == null ? now : this.uploadOrigin + this.uploadSlot * this.period);
    if (now < deadline) { this.uploadTimer = this.setTimer(() => this.dispatch(epoch), Math.ceil(deadline - now)); return; }
    const frame = this.pending; this.pending = null; this.uploading = true;
    if (this.uploadOrigin == null) { this.uploadOrigin = now; this.uploadSlot = 1; }
    else this.uploadSlot = Math.max(this.uploadSlot, Math.floor((now - this.uploadOrigin) / this.period)) + 1;
    this.send(frame, now, epoch);
  }
  async send(frame, uploadStartedAt, epoch) {
    const controller = this.makeController(); this.controller = controller;
    let rejectCancelled;
    const cancelled = new Promise((resolve, reject) => { rejectCancelled = reject; });
    const cancel = () => rejectCancelled(new DOMException('影格傳送已停止。', 'AbortError'));
    this.cancelUpload = cancel;
    this.uploadTimeout = this.setTimer(() => { controller.abort(); rejectCancelled(new DOMException('影格傳送逾時。', 'TimeoutError')); }, this.timeoutMs);
    try {
      await Promise.race([Promise.resolve().then(() => { if (!this.isCurrent(epoch)) throw new DOMException('影格傳送已停止。', 'AbortError'); return this.upload(frame.value, controller.signal); }), cancelled]);
      if (this.isCurrent(epoch)) this.onAccepted(frame.value, { captureStartedAt: frame.captureStartedAt, captureCompletedAt: frame.captureCompletedAt, uploadStartedAt, completedAt: this.now() });
    } catch (error) {
      if (!this.isCurrent(epoch)) return;
      if (error?.status === 429) {
        const seconds = Number(error.retryAfter);
        this.discardCapturedBefore = this.now();
        this.backoffUntil = this.now() + Math.max(1000, (Number.isFinite(seconds) && seconds > 0 ? seconds : 1) * 1000);
        if (this.pending) this.dropped += 1;
        this.pending = null; this.dropped += 1;
        this.clearTimer(this.uploadTimer); this.uploadTimer = null;
      } else this.fail(error);
    } finally {
      this.clearTimer(this.uploadTimeout); this.uploadTimeout = null;
      if (this.controller === controller) this.controller = null;
      if (this.cancelUpload === cancel) this.cancelUpload = null;
      this.uploading = false;
      if (this.isCurrent(epoch)) this.dispatch(epoch);
    }
  }
  fail(error) { this.stop(); this.onError(error); }
}
