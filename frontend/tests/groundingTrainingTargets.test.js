import test from 'node:test';
import assert from 'node:assert/strict';
import { applyGroundingTrainingTargets, groundingParameterVisible, groundingTrainingTargets, selectInferenceCheckpoint } from '../src/groundingTrainingTargets.js';

const modelId = 'grounding_dino_base';
const chunked = { id: 'chunked', training: {
  classes: Array.from({ length: 170 }, (_, index) => `class_${index}`),
  caption_mode: 'chunked', caption: '',
  caption_chunks: [{ caption: 'first group only .', global_class_ids: Array.from({ length: 90 }, (_, i) => i), text_tokens: 350 },
    { caption: 'remaining classes .', global_class_ids: Array.from({ length: 80 }, (_, i) => i + 90), text_tokens: 310 }],
} };
const normal = { id: 'normal', training: { classes: ['person', 'meal'], caption: 'person . meal .' } };

test('chunked checkpoint exposes all 170 classes and 2 groups without treating first caption as complete', () => {
  const targets = groundingTrainingTargets(modelId, chunked);
  assert.equal(targets.classCount, 170);
  assert.equal(targets.chunkCount, 2);
  assert.equal(targets.chunked, true);
  assert.equal(targets.prompt, '');
  const params = { prompt: 'my custom target .', conf: 0.3, use_training_classes: false };
  assert.deepEqual(applyGroundingTrainingTargets(params, targets), { ...params, use_training_classes: true });
  assert.equal(params.use_training_classes, false);
});

test('choosing chunked weights enables full targets while base, ordinary and unknown weights reset the mode', () => {
  const data = { modelId, params: { prompt: 'custom .', conf: 0.25 }, checkpointJobId: 'old', jobId: 'old-result', runMediaId: 'old-image' };
  const selected = selectInferenceCheckpoint(data, chunked.id, [chunked, normal]);
  assert.equal(selected.params.use_training_classes, true);
  assert.equal(selected.params.prompt, 'custom .');
  assert.equal(selected.checkpointJobId, '');
  assert.equal(selected.jobId, null);
  assert.equal(selected.runMediaId, null);
  for (const id of ['', normal.id, 'unknown']) {
    assert.equal(selectInferenceCheckpoint({ ...data, ...selected }, id, [chunked, normal]).params.use_training_classes, false);
  }
  assert.equal(data.params.use_training_classes, undefined);
});

test('normal checkpoint retains manual apply behaviour and preserves unrelated parameters', () => {
  const targets = groundingTrainingTargets(modelId, normal);
  assert.deepEqual(applyGroundingTrainingTargets({ conf: 0.2, use_training_classes: true }, targets), {
    conf: 0.2, use_training_classes: false, prompt: 'person . meal .',
  });
  assert.equal(targets.chunkCount, 1);
});

test('complete saved targets hide custom prompt until unchecked; base weights still show custom prompt', () => {
  const prompt = { key: 'prompt' };
  assert.equal(groundingParameterVisible(prompt, modelId, chunked, { use_training_classes: true }), false);
  assert.equal(groundingParameterVisible(prompt, modelId, chunked, { use_training_classes: false }), true);
  assert.equal(groundingParameterVisible(prompt, modelId, undefined, { use_training_classes: true }), true);
  assert.equal(groundingParameterVisible({ key: 'conf' }, modelId, chunked, { use_training_classes: true }), true);
  assert.equal(groundingParameterVisible({ key: 'use_training_classes' }, modelId, undefined, {}), false);
});

test('full training class scores do not expose an unused text threshold; custom targets restore it', () => {
  const textThreshold = { key: 'text_threshold' };
  for (const checkpoint of [chunked, normal]) {
    assert.equal(groundingParameterVisible(textThreshold, modelId, checkpoint, { use_training_classes: true }), false);
    assert.equal(groundingParameterVisible(textThreshold, modelId, checkpoint, { use_training_classes: false }), true);
  }
  assert.equal(groundingParameterVisible(textThreshold, modelId, undefined, {}), true);
});

test('other models never receive Grounding-specific targets or parameter flags', () => {
  assert.equal(groundingTrainingTargets('yolo26x', chunked), null);
  assert.equal(groundingTrainingTargets(modelId, undefined), null);
  const patch = selectInferenceCheckpoint({ modelId: 'yolo26x', params: { conf: 0.25 } }, 'new', []);
  assert.equal('params' in patch, false);
});
