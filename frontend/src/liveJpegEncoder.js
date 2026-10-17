// Reuse the drawing surface and prefer promise-based OffscreenCanvas encoding.
// There is one encode in flight and no frame queue; disposal invalidates its result.
export class LiveJpegEncoder {
  constructor({ getFallbackCanvas, OffscreenCanvasClass = globalThis.OffscreenCanvas, maxWidth = 960, maxHeight = 720, quality = 0.76, maxBytes = 1024 * 1024 } = {}) {
    Object.assign(this, { getFallbackCanvas, OffscreenCanvasClass, maxWidth, maxHeight, quality, maxBytes });
    this.surface = null; this.context = null; this.offscreen = false; this.disabledOffscreen = false;
    this.busy = false; this.disposed = false;
  }
  assertActive() { if (this.disposed) throw new DOMException('影格轉換已停止。', 'AbortError'); }
  dispose() { this.disposed = true; this.surface = null; this.context = null; }
  getSurface() {
    this.assertActive();
    if (this.surface) return this.surface;
    if (!this.disabledOffscreen && typeof this.OffscreenCanvasClass === 'function') {
      try {
        const surface = new this.OffscreenCanvasClass(1, 1);
        if (typeof surface.convertToBlob === 'function') {
          const context = surface.getContext('2d', { alpha: false });
          if (context) { this.surface = surface; this.context = context; this.offscreen = true; return surface; }
        }
      } catch { /* Use the browser's HTML canvas when OffscreenCanvas is unavailable. */ }
    }
    const surface = this.getFallbackCanvas?.(), context = surface?.getContext('2d', { alpha: false });
    if (!surface || !context || typeof surface.toBlob !== 'function') throw new Error('瀏覽器沒有提供可用的 JPEG 影格轉換介面。');
    this.surface = surface; this.context = context; this.offscreen = false;
    return surface;
  }
  async drawAndEncode(video, width, height) {
    const surface = this.getSurface();
    if (surface.width !== width) surface.width = width;
    if (surface.height !== height) surface.height = height;
    this.context.drawImage(video, 0, 0, width, height);
    if (this.offscreen) return surface.convertToBlob({ type: 'image/jpeg', quality: this.quality });
    return new Promise((resolve, reject) => {
      try { surface.toBlob(resolve, 'image/jpeg', this.quality); } catch (error) { reject(error); }
    });
  }
  async encode(video) {
    this.assertActive();
    if (this.busy) throw new DOMException('上一個影格仍在轉換。', 'InvalidStateError');
    const sourceWidth = video?.videoWidth, sourceHeight = video?.videoHeight;
    if (!(sourceWidth > 0) || !(sourceHeight > 0) || !Number.isFinite(sourceWidth) || !Number.isFinite(sourceHeight) || video.readyState < 2) throw new Error('攝影機沒有提供可讀取的影格，請停止後重新開啟。');
    const scale = Math.min(this.maxWidth / sourceWidth, this.maxHeight / sourceHeight, 1);
    const width = Math.max(1, Math.floor(sourceWidth * scale)), height = Math.max(1, Math.floor(sourceHeight * scale));
    this.busy = true;
    try {
      let blob;
      try { blob = await this.drawAndEncode(video, width, height); }
      catch (error) {
        this.assertActive();
        if (!this.offscreen || error?.name === 'SecurityError') throw error;
        // An implemented but unsupported encoder is retried once, then the
        // same HTML surface is reused for the remainder of this session.
        this.disabledOffscreen = true; this.surface = null; this.context = null;
        blob = await this.drawAndEncode(video, width, height);
      }
      this.assertActive();
      if (!blob || blob.type !== 'image/jpeg' || blob.size < 1 || blob.size > this.maxBytes) throw new Error('攝影機影格無法轉換成 1 MB 以下的 JPEG，請改用較低解析度的來源。');
      return blob;
    } finally { this.busy = false; }
  }
}
