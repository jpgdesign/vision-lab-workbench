import test from 'node:test';
import assert from 'node:assert/strict';
import { completedTrainingSignature, mediaTrainingUsage, mergeMediaUsageSnapshot } from '../src/mediaTrainingUsage.js';

test('training mark requires backend confirmed usage, never filename or job IDs alone', () => {
  for (const media of [{ name: 'train_bento.jpg' }, { name: '已訓練.jpg', training_usage: { used: false } },
    { name: 'image.jpg', training_usage: { used: 'true', job_ids: ['a'] } }, { name: 'image.jpg', training_usage: { job_ids: ['a'] } }]) {
    assert.equal(mediaTrainingUsage(media).used, false);
    assert.equal(mediaTrainingUsage(media).accessibleLabel, media.name);
  }
});

test('confirmed training usage adds accessible label and local completed scope with unique job count', () => {
  const usage = mediaTrainingUsage({ name: 'bento.jpg', training_usage: { used: true, job_ids: ['a', 'a', 'b', null] } });
  assert.equal(usage.used, true); assert.equal(usage.accessibleLabel, 'bento.jpg · 已用於訓練');
  assert.match(usage.title, /本機已完成/); assert.match(usage.title, /2 個工作/); assert.match(usage.title, /不限目前選用模型/);
  assert.match(mediaTrainingUsage({ name: 'val.jpg', training_usage: { used: false } }).title, /不代表未曾/);
});

test('completed signature detects jobs that finish between polls but ignores inference and active training', () => {
  const initial = [{ id: 'previous', kind: 'training', status: 'completed' }];
  const signature = completedTrainingSignature(initial);
  assert.equal(completedTrainingSignature([...initial, { id: 'active', kind: 'training', status: 'running' }, { id: 'inference', kind: 'inference', status: 'completed' }]), signature);
  const completed = [...initial, { id: 'fast', kind: 'training', status: 'completed' }];
  assert.notEqual(completedTrainingSignature(completed), signature);
  assert.equal(completedTrainingSignature([...completed].reverse()), completedTrainingSignature(completed));
});

test('usage refresh updates known metadata while preserving an upload completed after the request began', () => {
  const upload = { id: 'new-upload', training_usage: { used: false } };
  const previous = [{ id: 'known', training_usage: { used: false } }, upload];
  const incoming = [{ id: 'known', training_usage: { used: true, job_ids: ['train'] } }];
  const result = mergeMediaUsageSnapshot(previous, incoming);
  assert.equal(result[0].training_usage.used, true); assert.equal(result[1], upload);
  assert.equal(previous[0].training_usage.used, false);
});
