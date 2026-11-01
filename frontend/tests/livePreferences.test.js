import test from 'node:test';
import assert from 'node:assert/strict';
import { CAPTURE_FPS_OPTIONS, DEFAULT_LIVE_PREFERENCES, LIVE_PREFERENCES_KEY, normalizeLivePreferences, readLivePreferences, saveLivePreferences } from '../src/livePreferences.js';
import { snapshotLiveWorkflow, workflowLiveRequest } from '../src/workflowLive.js';

function storage(initial = null) {
  const values = new Map(initial == null ? [] : [[LIVE_PREFERENCES_KEY, initial]]);
  return { values, getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value) };
}

test('camera and VDO source settings survive reopening without creating sessions or connections', () => {
  const saved = storage();
  assert.equal(saveLivePreferences({ sourceMode: 'vdo', vdoUrl: 'https://vdo.ninja/?view=teaching_demo', vdoMode: 'iframe',
    deviceId: 'browser-device-id', captureFps: 24, mirror: true, durationMode: 'unlimited', durationValue: '48', durationUnit: 'hours' }, saved), true);
  const read = readLivePreferences(undefined, saved);
  assert.deepEqual(read, { version: 1, sourceMode: 'vdo', vdoUrl: 'https://vdo.ninja/?view=teaching_demo', vdoMode: 'iframe',
    deviceId: 'browser-device-id', captureFps: 24, mirror: true, durationMode: 'unlimited', durationValue: '48', durationUnit: 'hours' });
  assert.equal(saved.values.size, 1);
  assert.equal(Object.hasOwn(read, 'session'), false);
});

test('explicit workflow source mode overrides saved mode but retains compatible settings', () => {
  const saved = storage(JSON.stringify({ sourceMode: 'vdo', vdoUrl: 'https://vdo.ninja/?view=demo', captureFps: 15 }));
  const camera = readLivePreferences('camera', saved);
  assert.equal(camera.sourceMode, 'camera');
  assert.equal(camera.vdoUrl, 'https://vdo.ninja/?view=demo');
  assert.equal(camera.captureFps, 15);
  assert.equal(readLivePreferences('unknown', saved).sourceMode, 'vdo');
});

test('invalid, unavailable, oversized or future storage safely returns detached defaults', () => {
  for (const value of ['{broken', '[]', 'null', JSON.stringify({ version: 99, sourceMode: 'vdo' }), 'x'.repeat(8193)]) {
    assert.deepEqual(readLivePreferences(undefined, storage(value)), DEFAULT_LIVE_PREFERENCES);
  }
  const broken = { getItem() { throw new Error('blocked'); }, setItem() { throw new Error('quota'); } };
  assert.deepEqual(readLivePreferences(undefined, broken), DEFAULT_LIVE_PREFERENCES);
  assert.equal(saveLivePreferences({}, broken), false);
  assert.equal(saveLivePreferences({}, null), false);
  const first = readLivePreferences(undefined, null); first.sourceMode = 'vdo';
  assert.equal(readLivePreferences(undefined, null).sourceMode, 'camera');
});

test('stored URLs are VDO view URLs only; unrelated commands and remote payloads are removed', () => {
  for (const vdoUrl of ['javascript:alert(1)', 'https://example.com/?view=test', 'https://vdo.ninja/?push=demo',
    'https://vdo.ninja/?view=demo&room=private', 'https://user:pass@vdo.ninja/?view=test', 'https://vdo.ninja/?view=a&view=b', 'x'.repeat(2049)]) {
    assert.equal(normalizeLivePreferences({ vdoUrl }).vdoUrl, '', vdoUrl.slice(0, 80));
  }
  const result = normalizeLivePreferences({ vdoUrl: 'https://vdo.ninja/?view=demo&password=local-only&salt=example&api_key=discard-me&iframetarget=https://example.com' });
  assert.equal(result.vdoUrl, 'https://vdo.ninja/?view=demo&password=local-only&salt=example');
});

test('invalid typed settings do not bypass supported FPS, modes or duration validation', () => {
  const invalid = normalizeLivePreferences({ sourceMode: 'remote', deviceId: 'bad\u0000id', vdoMode: 'server', captureFps: 100,
    mirror: 'true', durationMode: 'forever', durationValue: '-1', durationUnit: 'days' });
  assert.deepEqual(invalid, DEFAULT_LIVE_PREFERENCES);
  assert.equal(normalizeLivePreferences({ captureFps: '30' }).captureFps, 30);
  assert.equal(normalizeLivePreferences({ deviceId: 'x'.repeat(1025) }).deviceId, '');
  for (const captureFps of CAPTURE_FPS_OPTIONS) assert.equal(normalizeLivePreferences({ captureFps }).captureFps, captureFps);
  assert.equal(normalizeLivePreferences({ durationValue: '9007199254740991', durationUnit: 'hours' }).durationValue, '15');
  assert.equal(normalizeLivePreferences({ durationValue: '0015', durationUnit: 'minutes' }).durationValue, '0015');
});

test('browser persistence whitelists fields and cannot collect workflow, model, key or session payloads', () => {
  const saved = storage();
  saveLivePreferences({ sourceMode: 'camera', api_key: 'must-not-save', authorization: 'must-not-save',
    key_file_path: 'must-not-save', graph: { secret: true }, models: ['private-model'], session: { id: 'private-job' } }, saved);
  const raw = saved.values.get(LIVE_PREFERENCES_KEY);
  assert.ok(!raw.includes('must-not-save') && !raw.includes('private-model') && !raw.includes('private-job'));
  assert.deepEqual(Object.keys(JSON.parse(raw)).sort(), Object.keys(DEFAULT_LIVE_PREFERENCES).sort());
});

test('workflow snapshots and live API requests never inherit local source URLs or device IDs', () => {
  const preferences = normalizeLivePreferences({ sourceMode: 'vdo', deviceId: 'private-device', vdoUrl: 'https://vdo.ninja/?view=private-view&password=private-password' });
  const graph = { version: 1, nodes: [{ id: 'source', type: 'source', params: {}, position: { x: 0, y: 0 } }], edges: [] };
  const snapshot = snapshotLiveWorkflow({ name: 'demo', graph, options: { device: 'cpu' }, labels: [], ...preferences });
  const request = workflowLiveRequest(snapshot, preferences.captureFps, null);
  for (const data of [snapshot, request]) {
    const text = JSON.stringify(data);
    assert.ok(!text.includes('private-device') && !text.includes('private-view') && !text.includes('private-password'));
  }
});
