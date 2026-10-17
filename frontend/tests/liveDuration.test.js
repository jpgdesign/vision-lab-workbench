import test from 'node:test';
import assert from 'node:assert/strict';
import { parseLiveDuration, formatLiveDuration } from '../src/liveDuration.js';

test('custom duration converts seconds, minutes, and hours without the old one-hour cap', () => {
  assert.equal(parseLiveDuration('1', 'seconds').seconds, 1);
  assert.equal(parseLiveDuration('15', 'minutes').seconds, 900);
  assert.equal(parseLiveDuration('2', 'hours').seconds, 7200);
  assert.equal(parseLiveDuration('48', 'hours').seconds, 172800);
  assert.equal(parseLiveDuration('0015', 'minutes').label, '15 分鐘');
});

test('unlimited explicitly produces null even if the retained custom input is invalid', () => {
  assert.deepEqual(parseLiveDuration('', 'minutes', 'unlimited'), { seconds: null, label: '無時限', error: '' });
  assert.equal(JSON.stringify({ max_duration_seconds: parseLiveDuration('', 'minutes', 'unlimited').seconds }), '{"max_duration_seconds":null}');
});

test('invalid custom values produce errors and no silently clamped payload', () => {
  for (const value of ['', ' ', '0', '-1', '1.5', 'Infinity', 'NaN', '1e3', '0x10']) {
    const result = parseLiveDuration(value, 'seconds');
    assert.ok(result.error, value);
    assert.equal(result.seconds, undefined, value);
  }
  assert.ok(parseLiveDuration('1', 'days').error);
  assert.ok(parseLiveDuration('1', 'seconds', 'unknown').error);
});

test('safe integers are accepted but overflow during unit conversion is rejected', () => {
  assert.equal(parseLiveDuration(String(Number.MAX_SAFE_INTEGER), 'seconds').seconds, Number.MAX_SAFE_INTEGER);
  assert.ok(parseLiveDuration(String(Number.MAX_SAFE_INTEGER + 1), 'seconds').error);
  assert.ok(parseLiveDuration(String(Number.MAX_SAFE_INTEGER), 'hours').error);
  assert.equal(parseLiveDuration('2501999792983', 'hours').seconds, 9007199254738800);
});

test('runtime summaries distinguish unlimited from missing limits and preserve exact seconds', () => {
  assert.equal(formatLiveDuration(null), '無時限');
  assert.equal(formatLiveDuration(undefined), '未設定');
  assert.equal(formatLiveDuration(900), '15 分鐘');
  assert.equal(formatLiveDuration(7200), '2 小時');
  assert.equal(formatLiveDuration(61), '61 秒');
});
