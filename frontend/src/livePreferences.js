import { parseLiveDuration } from './liveDuration.js';
import { parseVdoViewUrl } from './vdoSource.js';

// Source credentials/settings are local browser preferences, never graph data.
export const LIVE_PREFERENCES_KEY = 'vision-lab-live-source-v1';
export const CAPTURE_FPS_OPTIONS = [1, 2, 4, 8, 10, 15, 24, 25, 30];
export const DEFAULT_LIVE_PREFERENCES = Object.freeze({
  version: 1, sourceMode: 'camera', deviceId: '', vdoUrl: '', vdoMode: 'direct',
  captureFps: 30, mirror: false, durationMode: 'custom', durationValue: '15', durationUnit: 'minutes',
});

function localViewUrl(value) {
  if (typeof value !== 'string' || value.length > 2048 || !value.trim()) return '';
  try {
    const parsed = new URL(parseVdoViewUrl(value).url);
    for (const key of [...parsed.searchParams.keys()]) {
      if (!['view', 'password', 'pass', 'pw', 'p', 'salt'].includes(key)) parsed.searchParams.delete(key);
    }
    return parsed.href;
  } catch { return ''; }
}

export function normalizeLivePreferences(value) {
  const defaults = { ...DEFAULT_LIVE_PREFERENCES };
  if (!value || typeof value !== 'object' || Array.isArray(value) || value.version != null && value.version !== 1) return defaults;
  const durationUnit = ['seconds', 'minutes', 'hours'].includes(value.durationUnit) ? value.durationUnit : defaults.durationUnit;
  const candidateDuration = typeof value.durationValue === 'string' ? value.durationValue.trim() : '';
  const validDuration = candidateDuration.length <= 16 && !parseLiveDuration(candidateDuration, durationUnit, 'custom').error;
  return {
    version: 1,
    sourceMode: ['camera', 'vdo'].includes(value.sourceMode) ? value.sourceMode : defaults.sourceMode,
    deviceId: typeof value.deviceId === 'string' && value.deviceId.length <= 1024 && !/[\u0000-\u001f]/.test(value.deviceId) ? value.deviceId : '',
    vdoUrl: localViewUrl(value.vdoUrl),
    vdoMode: ['direct', 'iframe'].includes(value.vdoMode) ? value.vdoMode : defaults.vdoMode,
    captureFps: CAPTURE_FPS_OPTIONS.includes(value.captureFps) ? value.captureFps : defaults.captureFps,
    mirror: typeof value.mirror === 'boolean' ? value.mirror : defaults.mirror,
    durationMode: ['custom', 'unlimited'].includes(value.durationMode) ? value.durationMode : defaults.durationMode,
    durationValue: validDuration ? candidateDuration : defaults.durationValue,
    durationUnit: validDuration ? durationUnit : defaults.durationUnit,
  };
}

function browserStorage() {
  try { return globalThis.localStorage; } catch { return null; }
}

export function readLivePreferences(sourceMode, storage = browserStorage()) {
  let value;
  try {
    const raw = storage?.getItem(LIVE_PREFERENCES_KEY);
    if (raw && raw.length <= 8192) value = JSON.parse(raw);
  } catch { /* Storage can be unavailable; explicit connection still works. */ }
  const preferences = normalizeLivePreferences(value);
  if (['camera', 'vdo'].includes(sourceMode)) preferences.sourceMode = sourceMode;
  return preferences;
}

export function saveLivePreferences(value, storage = browserStorage()) {
  try {
    if (!storage) return false;
    storage.setItem(LIVE_PREFERENCES_KEY, JSON.stringify(normalizeLivePreferences(value)));
    return true;
  } catch { return false; }
}
