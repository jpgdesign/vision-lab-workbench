import test from 'node:test';
import assert from 'node:assert/strict';
import { chooseFamilyModel, modelFamilies, modelFamilyKey, modelReadiness, switchedModelPatch } from '../src/modelTasks.js';
import { checkpointRequest } from '../src/checkpoints.js';

const model = (id, task, family_id, available = true) => ({
  id, name: id, task, family_id, family_name: family_id, available,
  inference_supported: true, capabilities: { inference_params: [{ key: 'conf', default: 0.25 }, { key: 'device', default: 'auto' }] },
});
const models = [
  model('yolov12x', 'detection', 'yolov12'), model('yolov12x_seg', 'segmentation', 'yolov12'),
  model('yolo26x', 'detection', 'yolo26'), model('yolo26x_seg', 'segmentation', 'yolo26'),
  model('rf_detr_2xl', 'detection', 'rf_detr'), model('rf_detr_seg_2xl', 'segmentation', 'rf_detr'),
  model('x3d_l', 'action'),
];

test('three detector families group task variants while standalone models stay separate', () => {
  const families = modelFamilies(models);
  assert.deepEqual(families.map(value => value.key), ['yolov12', 'yolo26', 'rf_detr', 'x3d_l']);
  assert.deepEqual(families.map(value => value.models.length), [2, 2, 2, 1]);
  assert.equal(modelFamilyKey(models.at(-1)), 'x3d_l');
  assert.equal(families[0].models[1].id, 'yolov12x_seg');
});

test('changing families preserves the current runnable task and otherwise falls back to detection', () => {
  assert.equal(chooseFamilyModel(models, 'rf_detr', 'segmentation').id, 'rf_detr_seg_2xl');
  assert.equal(chooseFamilyModel(models, 'yolov12', 'action').id, 'yolov12x');
  assert.equal(chooseFamilyModel(models, 'x3d_l', 'segmentation').id, 'x3d_l');
  assert.equal(chooseFamilyModel(models, 'missing', 'segmentation'), undefined);
});

test('unavailable variants remain discoverable and show real readiness', () => {
  const choices = models.map(value => value.id === 'rf_detr_seg_2xl' ? { ...value, available: false } : value);
  assert.equal(chooseFamilyModel(choices, 'rf_detr', 'segmentation').id, 'rf_detr_2xl');
  const segmentation = modelFamilies(choices).find(value => value.key === 'rf_detr').models.find(value => value.task === 'segmentation');
  assert.equal(segmentation.id, 'rf_detr_seg_2xl');
  assert.equal(modelReadiness(segmentation), '未就緒');
  assert.equal(modelReadiness({ ...segmentation, inference_supported: false }), '推論尚未接入');
  assert.equal(modelReadiness(models[0]), '');
});

test('switching task clears detection checkpoint, result, source and incompatible parameters', () => {
  const oldCard = { id: 'keep-card', mediaId: 'keep-input', widthMode: 'manual', modelId: 'yolo26x', params: { conf: 0.9, prompt: 'person', accept_platform_license: true }, checkpointId: 'detect-checkpoint', checkpointJobId: 'detect-training', jobId: 'detect-result', runMediaId: 'old-input' };
  const next = { ...oldCard, ...switchedModelPatch(models.find(value => value.id === 'yolo26x_seg')) };
  assert.equal(next.modelId, 'yolo26x_seg');
  assert.deepEqual(next.params, { conf: 0.25, device: 'auto' });
  assert.deepEqual(checkpointRequest(next), {});
  assert.equal(next.jobId, null);
  assert.equal(next.runMediaId, null);
  assert.equal(next.id, 'keep-card');
  assert.equal(next.mediaId, 'keep-input');
  assert.equal(next.widthMode, 'manual');
  assert.equal(oldCard.checkpointId, 'detect-checkpoint');
});

test('switching back to detection does not reuse segmentation checkpoint or license flags', () => {
  const next = { checkpointId: 'segment-checkpoint', checkpointJobId: 'segment-training', ...switchedModelPatch(models.find(value => value.id === 'rf_detr_2xl')) };
  assert.deepEqual(checkpointRequest(next), {});
  assert.equal(next.modelId, 'rf_detr_2xl');
  assert.equal(Object.hasOwn(next.params, 'accept_platform_license'), false);
  assert.equal(switchedModelPatch(undefined), null);
});
