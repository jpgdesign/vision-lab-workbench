import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizedPoint, parseWorkflowGraph, polygonError, workflowCatalogGroups, workflowConnectionError, workflowDefaults, workflowDraftNodeStats, workflowFromFlow, workflowGraphErrors, workflowParameterValue, workflowPortLabel, workflowRemoteModels, workflowRemoteNode, workflowTemplateProfile, workflowToFlow } from '../src/workflowGraph.js';

const catalog = [
  { type: 'source', label: '來源', inputs: [], outputs: [{ name: 'frame', type: 'frame' }] },
  { type: 'detector', label: '偵測', inputs: [{ name: 'frame', type: 'frame', required: true }], outputs: [{ name: 'detections', type: 'detections' }], params: [{ key: 'model_id', default: 'yolo26x' }, { key: 'labels', default: ['person'] }] },
  { type: 'tracker', label: '追蹤', inputs: [{ name: 'observations', type: 'observations', accepts: ['detections', 'poses'], required: true }], outputs: [{ name: 'tracks', type: 'tracks' }] },
  { type: 'output', label: '輸出', inputs: [{ name: 'frame', type: 'frame', required: true }, { name: 'observations', type: 'observations', accepts: ['detections', 'poses', 'tracks'], multiple: true }], outputs: [] },
];
const nodes = ['source', 'detector', 'tracker', 'output'].map((type, index) => ({ id: type, type, position: { x: index * 280, y: 40 }, params: {} }));
const edge = (source, sourceHandle, target, targetHandle) => ({ id: `${source}-${target}`, source, sourceHandle, target, targetHandle });
const validGraph = { version: 1, nodes, edges: [edge('source', 'frame', 'detector', 'frame'), edge('detector', 'detections', 'tracker', 'observations'), edge('source', 'frame', 'output', 'frame'), edge('tracker', 'tracks', 'output', 'observations')] };

test('graph import/export round trip preserves typed handles and strips view state and runtime results', () => {
  const flow = workflowToFlow(parseWorkflowGraph({ name: 'Application', graph: validGraph }));
  flow.nodes[1].selected = true;
  flow.nodes[1].data.stat = { predictions: ['not exported'] };
  assert.deepEqual(workflowFromFlow(flow.nodes, flow.edges), validGraph);
  assert.deepEqual(workflowGraphErrors(validGraph, catalog), []);
});

test('connections accept union types and multiple observations but reject mismatch, duplicate and occupied single inputs', () => {
  const empty = { ...validGraph, edges: [] };
  assert.equal(workflowConnectionError(edge('detector', 'detections', 'tracker', 'observations'), empty, catalog), '');
  assert.equal(workflowConnectionError(edge('detector', 'detections', 'output', 'observations'), validGraph, catalog), '');
  assert.match(workflowConnectionError(edge('source', 'frame', 'tracker', 'observations'), empty, catalog), /類型不同/);
  assert.match(workflowConnectionError(validGraph.edges[0], validGraph, catalog), /已經存在/);
  assert.match(workflowConnectionError(edge('detector', 'detections', 'tracker', 'observations'), { ...validGraph, edges: [edge('different-detector', 'detections', 'tracker', 'observations')] }, catalog), /只能連接一個/);
  assert.match(workflowConnectionError({ source: 'source', target: 'detector' }, empty, catalog), /有效/);
});

test('cycles cannot be introduced even when both handles have a compatible type', () => {
  const cyclicCatalog = [{ type: 'processor', inputs: [{ name: 'in', type: 'frame' }], outputs: [{ name: 'out', type: 'frame' }] }];
  const graph = { version: 1, nodes: ['a', 'b', 'c'].map(id => ({ id, type: 'processor', params: {}, position: { x: 0, y: 0 } })), edges: [edge('a', 'out', 'b', 'in'), edge('b', 'out', 'c', 'in')] };
  assert.match(workflowConnectionError(edge('c', 'out', 'a', 'in'), graph, cyclicCatalog), /循環/);
  assert.match(workflowConnectionError(edge('a', 'out', 'a', 'in'), graph, cyclicCatalog), /自己/);
});

test('missing required inputs and unknown nodes remain visible as actionable validation errors', () => {
  assert.match(workflowGraphErrors({ ...validGraph, edges: [] }, catalog).join(' '), /偵測.*尚未連接/);
  assert.match(workflowGraphErrors(validGraph, catalog.filter(spec => spec.type !== 'tracker')).join(' '), /不支援的節點/);
  assert.match(workflowGraphErrors({ version: 1, nodes: [], edges: [] }, catalog).join(' '), /新增至少一個/);
});

test('invalid graph references, duplicate identifiers and non-finite positions are rejected on import', () => {
  assert.throws(() => parseWorkflowGraph({ ...validGraph, nodes: [...nodes, nodes[0]] }), /唯一/);
  assert.throws(() => parseWorkflowGraph({ ...validGraph, edges: [edge('missing', 'out', 'output', 'frame')] }), /不存在/);
  assert.throws(() => parseWorkflowGraph({ ...validGraph, nodes: [{ ...nodes[0], position: { x: Infinity, y: 1 } }] }), /有限/);
  assert.throws(() => parseWorkflowGraph({ version: 2, nodes, edges: [] }), /格式不正確/);
});

test('credentials and local key-file paths cannot enter exported graph parameters even when nested', () => {
  for (const key of ['api_key', 'apiKey', 'access_token', 'authorization', 'key_file_path', 'password']) {
    assert.throws(() => workflowParameterValue({ nested: [{ [key]: 'secret' }] }), /金鑰/);
    assert.throws(() => parseWorkflowGraph({ ...validGraph, nodes: [{ ...nodes[0], params: { [key]: 'secret' } }], edges: [] }), /金鑰/);
  }
  assert.deepEqual(workflowParameterValue({ model_id: 'api-model', instruction: 'Describe actions', max_calls: 10 }), { model_id: 'api-model', instruction: 'Describe actions', max_calls: 10 });
});

test('node defaults are independent copies so editing one labels or zone array cannot change another node', () => {
  const first = workflowDefaults(catalog[1]), second = workflowDefaults(catalog[1]);
  first.labels.push('helmet');
  assert.deepEqual(second.labels, ['person']);
  assert.deepEqual(catalog[1].params[1].default, ['person']);
});

test('zone editing maps pixel pointers to normalized image coordinates and rejects degenerate polygons', () => {
  assert.deepEqual(normalizedPoint(150, 75, { left: 50, top: 25, width: 200, height: 100 }), [0.5, 0.5]);
  assert.deepEqual(normalizedPoint(-100, 999, { left: 50, top: 25, width: 200, height: 100 }), [0, 1]);
  assert.equal(normalizedPoint(1, 1, { left: 0, top: 0, width: 0, height: 10 }), null);
  assert.equal(polygonError([[0, 0], [1, 0], [1, 1], [0, 1]]), '');
  assert.match(polygonError([[0, 0], [0.5, 0.5], [1, 1]]), /面積/);
  assert.match(polygonError([[0, 0], [1.1, 0], [1, 1]]), /0 到 1/);
});

test('only the actual remote VLM is marked cloud; local RelateAnything stays local', () => {
  assert.equal(workflowRemoteNode({ type: 'relation', category: '模型', label: 'RelateAnything' }), false);
  assert.equal(workflowRemoteNode({ type: 'vlm', category: '模型' }), true);
  assert.equal(workflowRemoteNode({ type: 'extension', remote: true }), true);
});

test('history sharing node IDs cannot attach another graph statistics to the current draft', () => {
  const job = { id: 'run-a', result: { node_stats: [{ id: 'detect', calls: 2, last_ms: 10 }] } };
  assert.deepEqual(workflowDraftNodeStats(job, { jobId: 'run-a', requestKey: 'graph-a' }, 'graph-a'), { detect: job.result.node_stats[0] });
  assert.deepEqual(workflowDraftNodeStats(job, { jobId: 'run-a', requestKey: 'graph-a' }, 'graph-with-edited-model'), {});
  assert.deepEqual(workflowDraftNodeStats({ ...job, id: 'history-b' }, { jobId: 'run-a', requestKey: 'graph-a' }, 'graph-a'), {});
  assert.deepEqual(workflowDraftNodeStats(job, null, 'graph-a'), {});
});

test('remote model choices come only from a working integration and reject malformed entries', () => {
  const integration = { available: true, models: ['vision-a', 'vision-b', 'vision-a', null, 4, ''] };
  assert.deepEqual(workflowRemoteModels(integration), ['vision-a', 'vision-b']);
  assert.deepEqual(workflowRemoteModels({ ...integration, available: false }), []);
  assert.deepEqual(workflowRemoteModels(null), []);
  assert.deepEqual(workflowRemoteModels({ available: true, models: 'vision-a' }), []);
});

test('node library searches category, labels, method names and parameters with category filtering', () => {
  const entries = [
    { type: 'cv_blur', category: 'OpenCV', label: '模糊', params: [{ key: 'method', label: '方法', options: ['gaussian', 'median'] }] },
    { type: 'cv_background', category: 'OpenCV', label: '背景減除', params: [{ key: 'method', options: [{ label: '高斯混合', value: 'MOG2' }] }] },
    { type: 'detector', category: '模型', label: '偵測' },
  ];
  assert.deepEqual(workflowCatalogGroups(entries, 'OpenCV gaussian').flatMap(group => group.nodes).map(node => node.type), ['cv_blur']);
  assert.equal(workflowCatalogGroups(entries, '高斯混合')[0].nodes[0].type, 'cv_background');
  assert.equal(workflowCatalogGroups(entries, 'mog2')[0].nodes[0].type, 'cv_background');
  assert.deepEqual(workflowCatalogGroups(entries, '', '模型').map(group => group.nodes.map(node => node.type)), [['detector']]);
  assert.deepEqual(workflowCatalogGroups(entries, 'unknown', 'OpenCV'), []);
  assert.deepEqual(workflowCatalogGroups(entries).map(group => group.category), ['OpenCV', '模型']);
});

test('port labels translate generic handles without changing custom labels or handle identifiers', () => {
  const frame = { name: 'frame', label: 'frame', type: 'frame' };
  assert.equal(workflowPortLabel(frame), '影像');
  assert.equal(frame.name, 'frame');
  assert.equal(workflowPortLabel({ ...frame, label: '處理後影像' }), '處理後影像');
  assert.equal(workflowPortLabel({ name: 'extension_port' }), 'extension_port');
});

test('OpenCV template badges distinguish model-free processing from CV plus learned models', () => {
  const template = types => ({ graph: { nodes: types.map(type => ({ type })) } });
  assert.deepEqual(workflowTemplateProfile(template(['source', 'cv_background', 'condition', 'output'])), { opencv: true, needsModel: false, label: 'OpenCV · 無需模型權重' });
  assert.deepEqual(workflowTemplateProfile(template(['source', 'cv_crop', 'detector', 'output'])), { opencv: true, needsModel: true, label: 'OpenCV ＋ 模型' });
  assert.equal(workflowTemplateProfile(template(['source', 'pose', 'output'])).opencv, false);
});
