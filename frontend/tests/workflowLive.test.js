import test from 'node:test';
import assert from 'node:assert/strict';
import { snapshotLiveWorkflow, workflowLiveRequest } from '../src/workflowLive.js';

const graph = () => ({ version: 1, nodes: [{ id: 'source', type: 'source', position: { x: 10, y: 20 }, params: {} }, { id: 'output', type: 'output', position: { x: 300, y: 20 }, params: {} }], edges: [{ id: 'e', source: 'source', sourceHandle: 'frame', target: 'output', targetHandle: 'frame' }] });

test('entering live work freezes a detached graph and options so later canvas changes cannot alter an active session', () => {
  const original = { name: 'Camera workflow', graph: graph(), options: { device: 'auto', frame_stride: 2, max_frames: 300 }, labels: ['來源', '輸出'] };
  const snapshot = snapshotLiveWorkflow(original);
  original.graph.nodes[0].position.x = 999;
  original.graph.nodes[1].params.modified = true;
  original.options.device = 'cpu';
  original.labels.push('new node');
  assert.equal(snapshot.graph.nodes[0].position.x, 10);
  assert.deepEqual(snapshot.graph.nodes[1].params, {});
  assert.equal(snapshot.options.device, 'auto');
  assert.deepEqual(snapshot.labels, ['來源', '輸出']);
});

test('live ingress request uses the exact workflow graph and capture duration without file-media or model-card state', () => {
  const snapshot = snapshotLiveWorkflow({ name: 'Camera', graph: graph(), options: { device: 'cuda:0', frame_stride: 1, max_frames: 300 }, labels: [] });
  const request = workflowLiveRequest(snapshot, 30, null);
  assert.deepEqual(Object.keys(request).sort(), ['capture_fps', 'graph', 'max_duration_seconds', 'options']);
  assert.equal(request.capture_fps, 30);
  assert.equal(request.max_duration_seconds, null);
  assert.equal(request.options.device, 'cuda:0');
  assert.deepEqual(request.graph, snapshot.graph);
  request.graph.nodes[0].position.x = 888;
  assert.equal(snapshot.graph.nodes[0].position.x, 10);
});
