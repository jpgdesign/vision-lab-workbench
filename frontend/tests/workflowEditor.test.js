import test from 'node:test';
import assert from 'node:assert/strict';
import { arrangeWorkflow, duplicateWorkflow, emptyWorkflowHistory, rememberWorkflow, sampleForTemplate, travelWorkflow, workflowKeyboardAction } from '../src/workflowEditor.js';

const node = (id, type = 'processor') => ({ id, type, params: { threshold: .5 }, position: { x: 10, y: 20 } });
const edge = (source, target) => ({ id: `${source}-${target}`, source, target, sourceHandle: 'frame', targetHandle: 'frame' });
const graph = { version: 1, nodes: [node('a', 'source'), node('b'), node('c', 'output')], edges: [edge('a', 'b'), edge('b', 'c')] };

test('undo and redo retain source selection, graph, options and independent parameter snapshots', () => {
  const first = { graph, name: 'first', mediaId: 'one', options: { frame_stride: 1 } };
  let history = rememberWorkflow(emptyWorkflowHistory(), first, '', 0);
  const second = structuredClone(first); second.graph.nodes[1].params.threshold = .8; second.mediaId = 'two';
  history = rememberWorkflow(history, second, '', 100);
  const third = { ...second, name: 'third' };
  const undo = travelWorkflow(history, third, 'undo');
  assert.equal(undo.snapshot.mediaId, 'two');
  assert.equal(undo.snapshot.name, 'first');
  const again = travelWorkflow(undo.history, undo.snapshot, 'undo');
  assert.equal(again.snapshot.graph.nodes[1].params.threshold, .5);
  const redo = travelWorkflow(again.history, again.snapshot, 'redo');
  assert.equal(redo.snapshot.graph.nodes[1].params.threshold, .8);
  redo.snapshot.graph.nodes[0].params.threshold = 1;
  assert.equal(first.graph.nodes[0].params.threshold, .5);
});

test('typing coalesces a parameter edit and a new edit after undo clears redo', () => {
  const first = { graph, name: 'A' }, next = { graph, name: 'B' };
  let history = rememberWorkflow(emptyWorkflowHistory(), first, 'field:threshold', 10);
  history = rememberWorkflow(history, next, 'field:threshold', 100);
  assert.equal(history.past.length, 1);
  const undo = travelWorkflow(history, next, 'undo');
  assert.equal(undo.snapshot.name, 'A');
  history = rememberWorkflow(undo.history, undo.snapshot, 'field:threshold', 150);
  assert.equal(history.future.length, 0);
  assert.equal(travelWorkflow(history, first, 'redo'), null);
});

test('history is bounded to forty actual undo steps', () => {
  let history = emptyWorkflowHistory();
  for (let i = 0; i < 60; i++) history = rememberWorkflow(history, { value: i }, '', i * 1000);
  assert.equal(history.past.length, 40);
  assert.equal(history.past[0].value, 20);
});

test('layout preserves graph semantics while placing branches in separate rows', () => {
  const branching = { ...graph, nodes: [...graph.nodes, node('d')], edges: [...graph.edges, edge('a', 'd'), edge('d', 'c')] };
  const laid = arrangeWorkflow(branching);
  const byId = Object.fromEntries(laid.nodes.map(n => [n.id, n]));
  assert.ok(byId.a.position.x < byId.b.position.x && byId.b.position.x < byId.c.position.x);
  assert.equal(byId.b.position.x, byId.d.position.x);
  assert.notEqual(byId.b.position.y, byId.d.position.y);
  assert.deepEqual(laid.edges, branching.edges);
  assert.deepEqual(graph.nodes[0].position, { x: 10, y: 20 });
});

test('duplicating selected nodes copies only internal wires and preserves source handles', () => {
  let id = 0;
  const copied = duplicateWorkflow(graph, ['a', 'b'], () => `copy-${++id}`);
  assert.equal(copied.graph.nodes.length, 5);
  assert.equal(copied.graph.edges.length, 3);
  assert.deepEqual(copied.selected, ['copy-1', 'copy-2']);
  assert.deepEqual(copied.graph.edges.at(-1), { ...edge('a', 'b'), id: 'copy-3', source: 'copy-1', target: 'copy-2' });
  copied.graph.nodes.at(-1).params.threshold = .2;
  assert.equal(graph.nodes[1].params.threshold, .5);
  assert.throws(() => duplicateWorkflow({ version: 1, nodes: Array.from({ length: 48 }, (_, i) => node(`n${i}`)), edges: [] }, ['n1'], () => 'new'), /48/);
});

test('editing text keeps native undo/delete and Tab remains accessible focus navigation', () => {
  const event = { key: 'z', ctrlKey: true, target: { tagName: 'INPUT' } };
  assert.equal(workflowKeyboardAction(event), null);
  assert.equal(workflowKeyboardAction({ ...event, target: { tagName: 'TEXTAREA' } }), null);
  assert.equal(workflowKeyboardAction({ ...event, target: { isContentEditable: true } }), null);
  assert.equal(workflowKeyboardAction({ ...event, target: {} }), 'undo');
  assert.equal(workflowKeyboardAction({ ...event, target: {}, shiftKey: true }), 'redo');
  assert.equal(workflowKeyboardAction({ key: 'Tab', target: {} }), null);
  assert.equal(workflowKeyboardAction({ key: 'n', target: {} }), 'add');
  assert.equal(workflowKeyboardAction({ key: 'Delete', target: {} }), 'delete');
});

test('sample picker accepts the exact API-resolved sample without an unrelated fallback', () => {
  const template = { sample: { available: true, media_id: 'example', title: 'Actual example', name: 'clip.mp4', kind: 'video', url: '/api/media/example' } };
  assert.equal(sampleForTemplate(template, [{ id: 'unrelated', name: 'photo.jpg' }]).media.id, 'example');
  const missing = sampleForTemplate({ sample: { ...template.sample, available: false, media_id: null, url: null } }, [{ id: 'unrelated' }]);
  assert.equal(missing.available, false);
  assert.equal(missing.media, null);
  assert.equal(sampleForTemplate(null), null);
});
