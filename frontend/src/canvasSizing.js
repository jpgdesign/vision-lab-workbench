import { applyNodeChanges } from '@xyflow/react';

export const MIN_NODE_WIDTH = 370;
export const DEFAULT_NODE_WIDTH = 420;
export const MAX_NODE_WIDTH = 900;
export const MIN_CANVAS_ZOOM = 0.1;

const NODE_GAP = 32;
const CANVAS_MARGIN = 56;
const MAX_MIN_HEIGHT = 2000;
const finite = value => typeof value === 'number' && Number.isFinite(value);
const clamp = (value, minimum, maximum) => Math.max(minimum, Math.min(maximum, value));

export function responsiveNodeWidth(viewportWidth) {
  if (!finite(viewportWidth) || viewportWidth <= 0) return DEFAULT_NODE_WIDTH;
  const columns = viewportWidth >= MIN_NODE_WIDTH * 2 + NODE_GAP + CANVAS_MARGIN ? 2 : 1;
  const available = (viewportWidth - CANVAS_MARGIN - (columns - 1) * NODE_GAP) / columns;
  return clamp(available, MIN_NODE_WIDTH, DEFAULT_NODE_WIDTH);
}

export function restoreNodeSizing(node = {}) {
  node = node && typeof node === 'object' ? node : {};
  // React Flow's explicit width takes precedence over style.width. Older saved
  // canvases can contain both after a manual resize.
  const previousWidth = finite(node.width) ? node.width : finite(node.style?.width) ? node.style.width : DEFAULT_NODE_WIDTH;
  const width = clamp(previousWidth, MIN_NODE_WIDTH, MAX_NODE_WIDTH);
  const storedMode = node.data?.widthMode;
  const widthMode = storedMode === 'auto' || storedMode === 'manual' ? storedMode : width === DEFAULT_NODE_WIDTH ? 'auto' : 'manual';
  const style = { width };
  if (finite(node.style?.minHeight)) style.minHeight = clamp(node.style.minHeight, 0, MAX_MIN_HEIGHT);
  // Do not migrate legacy fixed heights into minimums: the old default height
  // could already clip a result, and should now follow its content naturally.
  return { style, widthMode };
}

export function applyCanvasNodeChanges(changes, nodes) {
  const manual = new Map();
  const measuredChanges = changes.map(change => {
    if (change.type !== 'dimensions' || !change.setAttributes || !change.dimensions) return change;
    const next = { ...manual.get(change.id) };
    if ((change.setAttributes === true || change.setAttributes === 'width') && finite(change.dimensions.width)) {
      next.width = clamp(change.dimensions.width, MIN_NODE_WIDTH, MAX_NODE_WIDTH);
    }
    if ((change.setAttributes === true || change.setAttributes === 'height') && finite(change.dimensions.height)) {
      next.minHeight = clamp(change.dimensions.height, 0, MAX_MIN_HEIGHT);
    }
    manual.set(change.id, next);
    // Keep measured dimensions and resizing state, but prevent React Flow from
    // imposing a hard height on the outer node wrapper.
    return { ...change, setAttributes: false };
  });
  return applyNodeChanges(measuredChanges, nodes).map(node => {
    const sizing = manual.get(node.id);
    const hasHardDimensions = node.width !== undefined || node.height !== undefined || node.initialHeight !== undefined || node.style?.height !== undefined;
    if (!sizing && !hasHardDimensions) return node;
    const { width, height, initialHeight, ...rest } = node;
    const { height: styleHeight, ...style } = node.style || {};
    // Retain a legacy manually resized width when clearing explicit attributes.
    if (finite(width)) style.width = clamp(width, MIN_NODE_WIDTH, MAX_NODE_WIDTH);
    if (sizing?.width !== undefined) style.width = sizing.width;
    if (sizing?.minHeight !== undefined) style.minHeight = sizing.minHeight;
    return {
      ...rest,
      style,
      ...(sizing?.width !== undefined ? { data: { ...node.data, widthMode: 'manual' } } : {}),
    };
  });
}
