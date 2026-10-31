import test from 'node:test';
import assert from 'node:assert/strict';
import { applyCanvasNodeChanges, DEFAULT_NODE_WIDTH, MAX_NODE_WIDTH, MIN_CANVAS_ZOOM, MIN_NODE_WIDTH, responsiveNodeWidth, restoreNodeSizing } from '../src/canvasSizing.js';

const sampleNode = (extra = {}) => ({ id: 'one', position: { x: 28, y: 30 }, style: { width: 420 }, data: { widthMode: 'auto', modelId: 'yolo26x' }, ...extra });

test('restoring a legacy canvas drops fixed heights and retains actual manually resized width', () => {
  assert.deepEqual(restoreNodeSizing({ style: { width: 420, height: 590 }, height: 640, initialHeight: 600 }), { style: { width: 420 }, widthMode: 'auto' });
  assert.deepEqual(restoreNodeSizing({ style: { width: 420, height: 590 }, width: 650, height: 880 }), { style: { width: 650 }, widthMode: 'manual' });
  assert.deepEqual(restoreNodeSizing({ style: { width: 560, minHeight: 720 }, data: { widthMode: 'auto' } }), { style: { width: 560, minHeight: 720 }, widthMode: 'auto' });
  assert.equal(restoreNodeSizing({ style: { width: 420 }, data: { widthMode: 'manual' } }).widthMode, 'manual');
});

test('restoring malformed or oversized sizing keeps finite bounded values only', () => {
  assert.deepEqual(restoreNodeSizing({ width: Infinity, style: { width: '600', minHeight: NaN }, data: { widthMode: 'invalid' } }), { style: { width: DEFAULT_NODE_WIDTH }, widthMode: 'auto' });
  assert.deepEqual(restoreNodeSizing({ style: { width: -10, minHeight: -2 } }), { style: { width: MIN_NODE_WIDTH, minHeight: 0 }, widthMode: 'manual' });
  assert.deepEqual(restoreNodeSizing({ style: { width: 3000, minHeight: 5000 } }), { style: { width: MAX_NODE_WIDTH, minHeight: 2000 }, widthMode: 'manual' });
  assert.deepEqual(restoreNodeSizing(), { style: { width: DEFAULT_NODE_WIDTH }, widthMode: 'auto' });
  assert.deepEqual(restoreNodeSizing(null), { style: { width: DEFAULT_NODE_WIDTH }, widthMode: 'auto' });
});

test('responsive default width fits one or two columns without growing beyond its default', () => {
  assert.equal(MIN_CANVAS_ZOOM, 0.1);
  assert.equal(responsiveNodeWidth(426), 370);
  assert.equal(responsiveNodeWidth(460), 404);
  assert.equal(responsiveNodeWidth(827), 420);
  assert.equal(responsiveNodeWidth(828), 370);
  assert.equal(responsiveNodeWidth(888), 400);
  assert.equal(responsiveNodeWidth(1800), 420);
  assert.equal(responsiveNodeWidth(200), 370);
  for (const value of [undefined, null, NaN, Infinity, 0]) assert.equal(responsiveNodeWidth(value), 420);
});

test('corner resize preserves position changes and saves a minimum instead of a hard height', () => {
  const input = sampleNode({ width: 420, height: 590, initialHeight: 590, style: { width: 420, height: 590, borderColor: 'green' } });
  const result = applyCanvasNodeChanges([
    { id: 'one', type: 'position', position: { x: 10, y: 20 } },
    { id: 'one', type: 'dimensions', dimensions: { width: 600, height: 750 }, resizing: true, setAttributes: true },
  ], [input])[0];
  assert.deepEqual(result.style, { width: 600, minHeight: 750, borderColor: 'green' });
  assert.deepEqual(result.measured, { width: 600, height: 750 });
  assert.deepEqual(result.position, { x: 10, y: 20 });
  assert.equal(result.resizing, true);
  assert.equal(result.data.widthMode, 'manual');
  for (const key of ['width', 'height', 'initialHeight']) assert.equal(Object.hasOwn(result, key), false);
  assert.equal(input.height, 590);
  assert.equal(input.style.height, 590);
  assert.equal(input.data.widthMode, 'auto');
});

test('width-only resize leaves automatic content height free to grow and shrink', () => {
  const result = applyCanvasNodeChanges([{ id: 'one', type: 'dimensions', setAttributes: 'width', resizing: true, dimensions: { width: 500, height: 900 } }], [sampleNode()])[0];
  assert.deepEqual(result.style, { width: 500 });
  assert.equal(result.data.widthMode, 'manual');
  assert.equal(Object.hasOwn(result, 'height'), false);
});

test('height-only resize preserves responsive width and existing node position', () => {
  const input = sampleNode();
  const result = applyCanvasNodeChanges([{ id: 'one', type: 'dimensions', setAttributes: 'height', resizing: true, dimensions: { width: 700, height: 820 } }], [input])[0];
  assert.deepEqual(result.style, { width: 420, minHeight: 820 });
  assert.equal(result.data.widthMode, 'auto');
  assert.deepEqual(result.position, input.position);
});

test('ordinary measurements can increase and decrease without becoming a minimum or changing width mode', () => {
  let node = sampleNode();
  for (const height of [590, 1050, 620, 430]) {
    node = applyCanvasNodeChanges([{ id: 'one', type: 'dimensions', dimensions: { width: 420, height } }], [node])[0];
    assert.deepEqual(node.measured, { width: 420, height });
    assert.deepEqual(node.style, { width: 420 });
    assert.equal(node.data.widthMode, 'auto');
    assert.deepEqual(node.position, { x: 28, y: 30 });
    assert.equal(Object.hasOwn(node, 'height'), false);
  }
});

test('resize end and later content measurements retain only the requested minimum', () => {
  let node = applyCanvasNodeChanges([{ id: 'one', type: 'dimensions', setAttributes: true, resizing: true, dimensions: { width: 560, height: 700 } }], [sampleNode()])[0];
  node = applyCanvasNodeChanges([{ id: 'one', type: 'dimensions', resizing: false, dimensions: { width: 560, height: 700 } }], [node])[0];
  assert.equal(node.resizing, false);
  for (const height of [1000, 700]) {
    node = applyCanvasNodeChanges([{ id: 'one', type: 'dimensions', dimensions: { width: 560, height } }], [node])[0];
    assert.equal(node.measured.height, height);
    assert.deepEqual(node.style, { width: 560, minHeight: 700 });
  }
});

test('non-sizing changes keep untouched node identity and do not override user layout', () => {
  const one = sampleNode(), two = sampleNode({ id: 'two', position: { x: 680, y: 200 } });
  const result = applyCanvasNodeChanges([{ id: 'one', type: 'select', selected: true }], [one, two]);
  assert.equal(result[1], two);
  assert.equal(result[0].selected, true);
  assert.deepEqual(result[0].position, one.position);
  assert.equal(result[0].data.widthMode, 'auto');
});
