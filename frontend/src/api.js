export async function api(path, options = {}) {
  const { body, ...rest } = options;
  const isForm = body instanceof FormData;
  const response = await fetch(`/api${path}`, { ...rest, headers: { ...(body && !isForm ? { 'Content-Type': 'application/json' } : {}), ...rest.headers }, body: body ? (isForm ? body : JSON.stringify(body)) : undefined });
  let result;
  try { result = await response.json(); } catch { throw new Error(response.ok ? '伺服器回傳無法解析的資料。' : `伺服器錯誤 ${response.status}`); }
  if (!response.ok) {
    const detail = result.detail || result.error || result.message;
    const error = new Error(typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map(v => v.msg || JSON.stringify(v)).join('；') : `要求失敗 (${response.status})`);
    error.status = response.status;
    error.retryAfter = response.headers.get('Retry-After');
    throw error;
  }
  return result;
}
export const activeJob = job => job && ['queued', 'running'].includes(job.status);
export const statusNames = { queued: '排隊中', running: '執行中', completed: '已完成', failed: '失敗', cancelled: '已取消' };
export const taskNames = { detection: '物件偵測', action: '行為辨識', embedding: '視覺表徵' };
export const byteSize = bytes => !bytes ? '—' : bytes >= 1e9 ? `${(bytes / 1e9).toFixed(2)} GB` : `${(bytes / 1e6).toFixed(1)} MB`;
export const modelDefaults = model => Object.fromEntries((model?.capabilities?.inference_params || []).map(param => [param.key, param.default]));
export function optionParts(option) { return typeof option === 'object' ? { value: option.value ?? option.id ?? option.key, label: option.label ?? option.name ?? option.value ?? option.id } : { value: option, label: option }; }
export const uid = () => globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`;
