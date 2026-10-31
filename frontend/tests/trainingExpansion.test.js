import test from 'node:test';
import assert from 'node:assert/strict';
import { canTrainWithWeights, classAliases, effectiveTrainingParams, freshExpansionPreview, inferenceOnlyCheckpoint, isExpansionMode, requiresReplay, retentionRows, trainingRequest } from '../src/trainingExpansion.js';
import { canRunWithWeights, checkpointChoices } from '../src/checkpoints.js';

test('standard training cannot accidentally submit stale replay or aliases from expansion', () => {
  const body = trainingRequest({ modelId: 'yolo26x', datasetId: 'new', replayDatasetId: 'old', aliases: { chicken: 'bird' }, params: { lr: 0.001 } });
  assert.deepEqual(body, { kind: 'training', model_id: 'yolo26x', dataset_id: 'new', training_mode: 'standard', params: { lr: 0.001 } });
});

test('expansion sends source checkpoint, replay data and explicit aliases at the request top level', () => {
  const body = trainingRequest({ modelId: 'x3d_l', datasetId: 'new', checkpointId: 'job:abc', mode: 'expand_classes', replayDatasetId: 'old', aliases: { walk: 'walking' }, params: { strategy: 'head_only' } });
  assert.equal(body.checkpoint_id, 'job:abc'); assert.equal(body.replay_dataset_id, 'old');
  assert.deepEqual(body.class_aliases, { walk: 'walking' }); assert.deepEqual(body.params, { strategy: 'head_only' });
  assert.equal(trainingRequest({ mode: 'expand_classes' }).replay_dataset_id, null);
});

test('aliases reject incomplete and duplicate rows instead of silently overriding mappings', () => {
  assert.throws(() => classAliases([{ source: 'bird', target: '' }]), /都必須填寫/);
  assert.throws(() => classAliases([{ source: ' bird ', target: 'animal' }, { source: 'bird', target: 'pet' }]), /重複/);
  const aliases = classAliases([{ source: '__proto__', target: 'class' }]);
  assert.equal(JSON.stringify(aliases), '{"__proto__":"class"}');
  assert.equal(Object.getPrototypeOf(aliases), Object.prototype);
});

test('preview validity is bound to all serialized request settings, not just dataset identity', () => {
  const body = trainingRequest({ modelId: 'yolo26x', datasetId: 'new', mode: 'expand_classes', replayDatasetId: 'old', params: { lr: 0.01 } });
  const signature = JSON.stringify(body), preview = { signature, result: { valid: true } };
  assert.equal(freshExpansionPreview(preview, signature), true);
  for (const change of [{ params: { lr: 0.001 } }, { replay_dataset_id: 'different' }, { checkpoint_id: 'job:new' }, { class_aliases: { car: 'truck' } }, { training_mode: 'standard' }]) {
    assert.equal(freshExpansionPreview(preview, JSON.stringify({ ...body, ...change })), false);
  }
  assert.equal(freshExpansionPreview({ signature, result: { valid: false } }, signature), false);
  assert.equal(freshExpansionPreview(null, signature), false);
});

test('training can start from a valid checkpoint even when original weights are absent', () => {
  const model = { training_supported: true, available: false, environment_available: true };
  const choices = [{ id: 'trained', valid: true }];
  assert.equal(canTrainWithWeights(model, 'trained', choices), true);
  assert.equal(canTrainWithWeights(model, '', choices), false);
  assert.equal(canTrainWithWeights(model, 'missing', choices), false);
  assert.equal(canTrainWithWeights(model, 'trained', [{ id: 'trained', valid: false }]), false);
  assert.equal(canTrainWithWeights({ ...model, environment_available: false }, 'trained', choices), false);
  assert.equal(canTrainWithWeights({ ...model, training_supported: false, available: true }, 'trained', choices), false);
});

test('retention distinguishes absent metrics from genuine zero scores and derives delta from same-task measurements', () => {
  const rows = retentionRows({ metric: 'map50_95', before: { old: { map50_95: 0.4 } }, after: { old: { map50_95: 0 }, new: { map50_95: 0.25 } }, delta: { old: 99 } });
  assert.equal(rows[0].after, 0); assert.equal(rows[0].delta, -0.4);
  assert.equal(rows[1].before, null); assert.equal(rows[1].delta, null); assert.equal(rows[1].after, 0.25);
  assert.equal(retentionRows({ metric: 'accuracy', before: { old: { accuracy: null } }, after: { old: { accuracy: 0.9 } } })[0].delta, null);
  assert.deepEqual(retentionRows({ metric: 'unsupported' }), []);
});

test('self-contained directory checkpoint is available in inference choices without a huge downloadable zip', () => {
  const jobs = [{ id: 'dino', kind: 'training', model_id: 'dinov3_7b', status: 'completed', created_at: '2026-01-01T00:00:00Z', result: { training: { checkpoint_directory: '/jobs/dino/best', classes: ['a', 'b'] } } }];
  assert.equal(checkpointChoices('dinov3_7b', [], jobs)[0].id, 'job:dino');
  assert.equal(checkpointChoices('dinov3_7b', [], [{ ...jobs[0], status: 'failed' }]).length, 0);
});

test('all three expansion modes preview, but only replay expansion requires old training data', () => {
  for (const mode of ['expand_frozen', 'expand_classes', 'dual_branch']) {
    assert.equal(isExpansionMode(mode), true);
    const request = trainingRequest({ modelId: 'yolo26x', datasetId: 'new', mode });
    assert.equal(request.training_mode, mode); assert.equal(request.replay_dataset_id, null);
    assert.deepEqual(request.class_aliases, {});
    assert.equal(requiresReplay(mode), mode === 'expand_classes');
  }
  assert.equal(isExpansionMode('standard'), false);
});

test('frozen expansion enforces full backbone freeze or head-only without destroying other modes settings', () => {
  const detection = { strategy: 'full', freeze: 3, lr: 0.003 };
  assert.deepEqual(effectiveTrainingParams('expand_frozen', 'detection', detection), { strategy: 'freeze_backbone', freeze: 0, lr: 0.003 });
  assert.deepEqual(detection, { strategy: 'full', freeze: 3, lr: 0.003 });
  assert.equal(effectiveTrainingParams('dual_branch', 'detection', detection), detection);
  assert.deepEqual(effectiveTrainingParams('expand_frozen', 'action', { strategy: 'full' }), { strategy: 'head_only' });
  assert.deepEqual(effectiveTrainingParams('expand_frozen', 'embedding', { strategy: 'full' }), { strategy: 'head_only' });
});

test('optional old evaluation and explicit Grounding prompt names remain distinct from dataset classes', () => {
  const request = trainingRequest({ modelId: 'grounding_dino_base', datasetId: 'new', mode: 'dual_branch', replayDatasetId: 'old-evaluation', baseClassNames: ['person', 'car'] });
  assert.equal(request.replay_dataset_id, 'old-evaluation');
  assert.deepEqual(request.base_class_names, ['person', 'car']);
  assert.equal(Object.hasOwn(trainingRequest({ mode: 'expand_frozen' }), 'base_class_names'), false);
  assert.equal(Object.hasOwn(trainingRequest({ mode: 'standard', baseClassNames: ['person'] }), 'base_class_names'), false);
});

test('bundle directory output is selectable for inference but never a new training source', () => {
  const model = { available: true, environment_available: true, training_supported: true, inference_supported: true };
  const job = { id: 'bundle', model_id: 'yolo26x', kind: 'training', status: 'completed', created_at: '2026-11-01T00:00:00Z', result: { checkpoint_is_directory: true, training: { checkpoint_format: 'frozen_branch_bundle', classes: ['old', 'new'] } } };
  const choices = checkpointChoices('yolo26x', [], [job]);
  assert.equal(choices.length, 1); assert.equal(inferenceOnlyCheckpoint(choices[0]), true);
  assert.equal(canTrainWithWeights(model, 'job:bundle', choices), false);
  assert.equal(canRunWithWeights(model, { checkpointId: 'job:bundle' }, choices), true);
});

test('missing old evaluation never becomes a measured zero or fabricated delta', () => {
  const rows = retentionRows({ metric: 'accuracy', before: { old: null }, after: { old: null, new: { accuracy: 0 } }, delta: { old: 0 } });
  assert.equal(rows[0].before, null); assert.equal(rows[0].after, null); assert.equal(rows[0].delta, null);
  assert.equal(rows[1].after, 0);
});
