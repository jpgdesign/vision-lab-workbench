import { parseWorkflowGraph } from './workflowGraph.js';

const clone = value => structuredClone(value);
export const emptyWorkflowHistory = () => ({ past: [], future: [], group: '', time: 0 });

export function rememberWorkflow(history, snapshot, group = '', time = Date.now()) {
  const sameGroup = group && history.group === group && time - history.time < 700;
  const duplicate = JSON.stringify(history.past.at(-1)) === JSON.stringify(snapshot);
  return { past: sameGroup || duplicate ? history.past : [...history.past, clone(snapshot)].slice(-40),
    future: [], group, time };
}

export function travelWorkflow(history, current, direction) {
  const undo = direction === 'undo';
  const from = undo ? history.past : history.future;
  if (!from.length) return null;
  return { snapshot: clone(from.at(-1)), history: {
    past: undo ? history.past.slice(0, -1) : [...history.past, clone(current)].slice(-40),
    future: undo ? [...history.future, clone(current)].slice(-40) : history.future.slice(0, -1), group: '', time: 0,
  } };
}

export function arrangeWorkflow(graph, catalog = []) {
  const result = parseWorkflowGraph(graph), depth = new Map(), pending = new Set(result.nodes.map(node => node.id));
  // Longest predecessor path preserves branches and remains deterministic.
  while (pending.size) {
    let progress = false;
    for (const id of [...pending]) {
      const parents = result.edges.filter(edge => edge.target === id).map(edge => edge.source);
      if (parents.some(parent => pending.has(parent))) continue;
      depth.set(id, parents.length ? Math.max(...parents.map(parent => depth.get(parent) || 0)) + 1 : 0);
      pending.delete(id); progress = true;
    }
    if (!progress) { for (const id of pending) depth.set(id, 0); break; }
  }
  const columns = new Map();
  for (const node of result.nodes) {
    const level = depth.get(node.id) || 0;
    if (!columns.has(level)) columns.set(level, []);
    columns.get(level).push(node);
  }
  for (const [level, nodes] of columns) {
    let y = 60;
    for (const node of nodes) {
      node.position = { x: 60 + level * 310, y };
      const spec = catalog.find(item => item.type === node.type);
      y += 210 + Math.max(spec?.inputs?.length || 0, spec?.outputs?.length || 0) * 28;
    }
  }
  return result;
}

export function duplicateWorkflow(graph, ids, createId) {
  const result = parseWorkflowGraph(graph), selected = result.nodes.filter(node => ids.includes(node.id));
  if (result.nodes.length + selected.length > 48) throw new Error('流程最多可放入 48 個節點。');
  const mapping = new Map(selected.map(node => [node.id, createId()]));
  result.nodes.push(...selected.map(node => ({ ...clone(node), id: mapping.get(node.id), position: { x: node.position.x + 45, y: node.position.y + 45 } })));
  result.edges.push(...result.edges.filter(edge => mapping.has(edge.source) && mapping.has(edge.target)).map(edge => ({ ...edge, id: createId(), source: mapping.get(edge.source), target: mapping.get(edge.target) })));
  if (result.edges.length > 128) throw new Error('流程最多可放入 128 條連線。');
  return { graph: result, selected: [...mapping.values()] };
}

export function workflowKeyboardAction(event) {
  const target = event.target;
  if (target?.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(target?.tagName || '') || target?.closest?.('[role="dialog"]')) return null;
  const modifier = event.ctrlKey || event.metaKey;
  const key = event.key.toLowerCase();
  if (modifier && key === 'z') return event.shiftKey ? 'redo' : 'undo';
  if (modifier && key === 'y') return 'redo';
  if (modifier && key === 'd') return 'duplicate';
  if (!modifier && (key === 'delete' || key === 'backspace')) return 'delete';
  if (!modifier && !event.altKey && key === 'f') return 'fit';
  if (!modifier && !event.altKey && key === 'n') return 'add';
  return null;
}

export function sampleForTemplate(template, media = []) {
  const sample = template?.sample;
  if (!sample) return null;
  const selected = sample.media_id ? media.find(item => item.id === sample.media_id) || (sample.available && sample.url ? { id: sample.media_id, name: sample.name || sample.title, kind: sample.kind, url: sample.url } : null) : null;
  return { ...sample, available: Boolean(sample.available && selected), media: selected };
}
