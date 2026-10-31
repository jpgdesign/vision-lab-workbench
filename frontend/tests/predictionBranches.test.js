import test from 'node:test';
import assert from 'node:assert/strict';
import { branchPredictionGroups } from '../src/predictionBranches.js';

test('explicit branch groups preserve independent scores and original branch ordering', () => {
  const original = [{ label: 'person', score: 0.9 }, { label: 'dog', score: 0.8 }], added = [{ label: 'food', score: 0.95 }];
  const groups = branchPredictionGroups({ prediction_groups: { new: added, original }, predictions: [] });
  assert.deepEqual(groups.map(group => group.key), ['original', 'new']);
  assert.deepEqual(groups[0].predictions, original); assert.deepEqual(groups[1].predictions, added);
  assert.equal(groups[0].predictions[0].score, 0.9); assert.equal(groups[1].predictions[0].score, 0.95);
});

test('flat branch-tagged predictions are separated without mixing untagged detections into original', () => {
  const result = { predictions: [{ label: 'food', branch: 'new', score: 0.8 }, { label: 'person', branch: 'original', score: 0.7 }, { label: 'unknown', score: 0.9 }] };
  assert.deepEqual(branchPredictionGroups(result).map(group => [group.key, group.predictions.map(value => value.label)]), [['original', ['person']], ['new', ['food']], ['other', ['unknown']]]);
});

test('empty explicit groups still identify branches but a standard result is never labelled dual branch', () => {
  assert.equal(branchPredictionGroups({ prediction_groups: { original: [], new: [] } }).length, 2);
  assert.deepEqual(branchPredictionGroups({ predictions: [{ label: 'person', score: 0.7 }] }), []);
  assert.deepEqual(branchPredictionGroups(null), []);
});

test('backend list format retains both source groups when one branch has no detections', () => {
  const added = [{ label: 'bento', score: 0.8, branch: 'new' }];
  const result = { predictions: added, prediction_groups: [
    { branch: 'new', predictions: added, score_semantics: 'independent', input_params: { conf: .25 } },
    { branch: 'original', predictions: [], score_semantics: 'independent', input_params: { conf: .25 } },
  ] };
  const groups = branchPredictionGroups(result);
  assert.deepEqual(groups.map(group => group.key), ['original', 'new']);
  assert.deepEqual(groups[0].predictions, []);
  assert.equal(groups[1].predictions[0].score, .8);
});

test('backend list format with no detections keeps two visible empty branches', () => {
  const groups = branchPredictionGroups({ predictions: [], prediction_groups: [
    { branch: 'original', predictions: [] }, { branch: 'new', predictions: [] },
  ] });
  assert.equal(groups.length, 2);
  assert.deepEqual(groups.map(group => group.predictions), [[], []]);
});
