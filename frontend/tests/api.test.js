import test from 'node:test';
import assert from 'node:assert/strict';
import { api, apiErrorMessage } from '../src/api.js';

test('structured training preflight errors preserve actionable dataset problems', () => {
  assert.equal(apiErrorMessage({ message: '訓練預檢未通過', errors: ['old_val 缺少 dog', 'train / val 重複'], plan: { classes: ['dog'] } }, 422), '訓練預檢未通過；old_val 缺少 dog；train / val 重複');
  assert.equal(apiErrorMessage([{ msg: 'Field required' }], 422), 'Field required');
  assert.equal(apiErrorMessage('權重不存在', 409), '權重不存在');
  assert.equal(apiErrorMessage(null, 500), '要求失敗 (500)');
});

test('API surfaces structured dataset revalidation failure after a previously valid preview', async t => {
  const detail = { message: '舊資料集重新驗證失敗', errors: ['val/ 不存在'] };
  t.mock.method(globalThis, 'fetch', async () => ({ ok: false, status: 422, json: async () => ({ detail }), headers: new Headers() }));
  await assert.rejects(api('/jobs', { method: 'POST', body: { kind: 'training' } }), error => {
    assert.equal(error.message, '舊資料集重新驗證失敗；val/ 不存在');
    assert.equal(error.status, 422); assert.deepEqual(error.detail, detail); return true;
  });
});
