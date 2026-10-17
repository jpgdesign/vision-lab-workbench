import test from 'node:test';
import assert from 'node:assert/strict';
import { canRunWithWeights, checkpointChoices, checkpointRequest, checkpointUnavailable, selectedCheckpoint } from '../src/checkpoints.js';
import { applySystemDevices, deviceAvailable, systemDeviceOptions } from '../src/systemDevices.js';

const job = { id: 'abc123', kind: 'training', model_id: 'yolo26x', status: 'completed', created_at: '2026-01-01T00:00:00Z', result: { checkpoint_url: '/weights/best.pt', training: { classes: ['part'] } } };

test('checkpoint list combines model-specific external and completed training weights without duplicates', () => {
  const entries = [{ id: 'job:abc123', model_id: 'yolo26x', source: 'training_job', job_id: 'abc123', valid: true, name: 'parts' }, { id: 'external-1', model_id: 'yolo26x', source: 'external', valid: true, name: 'external parts' }, { id: 'external-other', model_id: 'rf_detr_2xl', source: 'external', valid: true }];
  const choices = checkpointChoices('yolo26x', entries, [job]);
  assert.equal(choices.length, 2); assert.equal(choices[0].name, 'parts');
  assert.deepEqual(choices[0].training.classes, ['part']); assert.equal(choices[1].source, 'external');
});

test('registry invalidation wins over a completed job so a missing file cannot be selected', () => {
  const choices = checkpointChoices('yolo26x', [{ id: 'job:abc123', model_id: 'yolo26x', source: 'training_job', job_id: 'abc123', valid: false }], [job]);
  assert.equal(choices.length, 1); assert.equal(checkpointUnavailable({ checkpointJobId: 'abc123' }, choices), true);
});

test('completed job fallback excludes unfinished, failed, wrong-model, or missing-checkpoint jobs', () => {
  const choices = checkpointChoices('yolo26x', [], [job, { ...job, id: 'running', status: 'running' }, { ...job, id: 'failed', status: 'failed' }, { ...job, id: 'wrong', model_id: 'other' }, { ...job, id: 'missing', result: {} }]);
  assert.deepEqual(choices.map(choice => choice.id), ['job:abc123']);
  assert.equal(checkpointUnavailable({ checkpointId: 'job:abc123' }, choices), false);
});

test('legacy and registry checkpoint requests are mutually exclusive and original weights omit both', () => {
  assert.equal(selectedCheckpoint({ checkpointJobId: 'abc123' }), 'job:abc123');
  assert.deepEqual(checkpointRequest({ checkpointJobId: 'abc123' }), { checkpoint_job_id: 'abc123' });
  assert.deepEqual(checkpointRequest({ checkpointId: 'external-1', checkpointJobId: 'old' }), { checkpoint_id: 'external-1' });
  assert.deepEqual(checkpointRequest({}), {});
  assert.equal(checkpointUnavailable({}, []), false);
  assert.equal(checkpointUnavailable({ checkpointId: 'deleted' }, []), true);
});

test('self-contained checkpoint can run without base weights only when the runtime is available', () => {
  const model = { inference_supported: true, available: false, environment_available: true };
  const selected = { checkpointId: 'external-1' }, choices = [{ id: 'external-1', valid: true }];
  assert.equal(canRunWithWeights(model, selected, choices), true);
  assert.equal(canRunWithWeights(model, {}, choices), false);
  assert.equal(canRunWithWeights({ ...model, environment_available: false }, selected, choices), false);
  assert.equal(canRunWithWeights(model, selected, [{ id: 'external-1', valid: false }]), false);
  assert.equal(canRunWithWeights({ ...model, available: true }, {}, []), true);
  assert.equal(canRunWithWeights({ ...model, inference_supported: false }, selected, choices), false);
});

test('GPU detection alone does not expose unusable CUDA devices', () => {
  const system = { nvidia: { detected: true, gpus: [{ index: 0, name: 'NVIDIA' }] }, torch: { cuda_available: false }, devices: [{ id: 'cpu' }] };
  assert.deepEqual(systemDeviceOptions(system).map(option => option.value), ['auto', 'cpu']);
  assert.equal(deviceAvailable('cuda:0', system), false);
  assert.equal(deviceAvailable('cpu', system), true);
});

test('usable multi-GPU options preserve exact CUDA indices and reject duplicates or unsupported device strings', () => {
  const system = { devices: [{ id: 'cpu' }, { id: 'cuda:0', name: 'first' }, { id: 'cuda:2', name: 'second' }, { id: 'cuda:2', name: 'duplicate' }, { id: 'mps' }, { id: 'cuda:bad' }] };
  assert.deepEqual(systemDeviceOptions(system).map(option => option.value), ['auto', 'cpu', 'cuda:0', 'cuda:2']);
  assert.equal(deviceAvailable('cuda:2', system), true); assert.equal(deviceAvailable('cuda:1', system), false);
});

test('device adapter updates inference and training choices without changing unrelated defaults or license consent', () => {
  const input = [{ id: 'model', license_accepted: false, capabilities: { inference_params: [{ key: 'device', default: 'cuda:0', options: ['cuda:0'] }, { key: 'conf', default: 0.5 }], training: { supported_params: ['device', 'epochs'], defaults: { device: 'cuda:0', epochs: 10, accept_platform_license: false } } } }];
  const original = structuredClone(input), output = applySystemDevices(input, { devices: [{ id: 'cpu' }] });
  assert.deepEqual(input, original); assert.equal(output[0].license_accepted, false);
  assert.equal(output[0].capabilities.inference_params[0].default, 'auto');
  assert.equal(output[0].capabilities.training.defaults.device, 'auto');
  assert.equal(output[0].capabilities.training.defaults.epochs, 10);
  assert.equal(output[0].capabilities.training.defaults.accept_platform_license, false);
  assert.deepEqual(output[0].capabilities.training.parameters[0].options.map(option => option.value), ['auto', 'cpu']);
});

test('unknown system metadata offers only auto and CPU without inventing hardware', () => {
  assert.deepEqual(systemDeviceOptions(null).map(option => option.value), ['auto', 'cpu']);
  assert.equal(deviceAvailable(undefined, null), true);
  assert.equal(deviceAvailable('cuda:0', null), false);
});
