export const WORKFLOW_STORAGE_KEY = 'vision-lab-applications-v1';
export const EMPTY_GRAPH = { version: 1, nodes: [], edges: [] };
const secretKey = /^(?:api[_-]?key|access[_-]?token|secret|authorization|key_file_path|password)$/i;
const plainObject = value => value && typeof value === 'object' && !Array.isArray(value);

function safeParameters(value, depth = 0) {
  if (depth > 20) throw new Error('節點參數的巢狀層級過深。');
  if (value === null || typeof value === 'boolean' || typeof value === 'string') return value;
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (Array.isArray(value)) return value.map(item => safeParameters(item, depth + 1));
  if (plainObject(value)) return Object.fromEntries(Object.entries(value).map(([key, item]) => {
    if (secretKey.test(key)) throw new Error('流程檔不可包含 API 金鑰、密碼或金鑰檔路徑；請在雲端連接設定中選擇本機金鑰檔。');
    if (['__proto__', 'constructor', 'prototype'].includes(key)) throw new Error('流程參數包含不支援的欄位。');
    return [key, safeParameters(item, depth + 1)];
  }));
  throw new Error('流程參數必須是可儲存的 JSON 值。');
}

export function parseWorkflowGraph(value) {
  const document = typeof value === 'string' ? JSON.parse(value) : value;
  const graph = document?.graph || document;
  if (!graph || graph.version !== 1 || !Array.isArray(graph.nodes) || !Array.isArray(graph.edges)) throw new Error('流程格式不正確：需要 version: 1、nodes 與 edges。');
  if (graph.nodes.length > 48 || graph.edges.length > 128) throw new Error('流程最多包含 48 個節點與 128 條連線。');
  const ids = new Set();
  const nodes = graph.nodes.map(node => {
    if (typeof node.id !== 'string' || !/^[A-Za-z0-9_-]{1,80}$/.test(node.id) || ids.has(node.id) || typeof node.type !== 'string' || !node.type) throw new Error('節點 ID 必須是唯一的英數字、底線或連字號，並指定節點類型。');
    ids.add(node.id);
    const position = node.position || {};
    if (![position.x, position.y].every(Number.isFinite)) throw new Error('節點位置必須是有限的座標數值。');
    if (!plainObject(node.params || {})) throw new Error('節點參數必須是 JSON 物件。');
    return { id: node.id, type: node.type, position: { x: position.x, y: position.y }, params: safeParameters(node.params || {}) };
  });
  const edgeIds = new Set();
  const edges = graph.edges.map((edge, index) => {
    if (!ids.has(edge.source) || !ids.has(edge.target)) throw new Error('連線參照了不存在的節點。');
    const id = edge.id || `edge-${index}`;
    if (edgeIds.has(id)) throw new Error('連線 ID 不可重複。');
    edgeIds.add(id);
    return { id, source: edge.source, target: edge.target, sourceHandle: edge.sourceHandle || null, targetHandle: edge.targetHandle || null };
  });
  return { version: 1, nodes, edges };
}

export function workflowFromFlow(nodes, edges) {
  return parseWorkflowGraph({ version: 1, nodes: nodes.map(node => ({ id: node.id, type: node.data.nodeType, position: node.position, params: node.data.params || {} })), edges });
}

export function workflowToFlow(graph) {
  return { nodes: graph.nodes.map(node => ({ id: node.id, type: 'workflow', position: node.position, data: { nodeType: node.type, params: node.params || {} } })), edges: graph.edges };
}

export function workflowDefaults(spec) {
  return Object.fromEntries((spec?.params || []).filter(param => param.default !== undefined).map(param => [param.key, structuredClone(param.default)]));
}

const port = (ports, handle) => ports?.find(item => item.name === handle);
const types = value => Array.isArray(value) ? value : String(value || '').split('|');
export function workflowConnectionError(connection, graph, catalog) {
  if (connection.source === connection.target) return '節點不能連接到自己。';
  const source = graph.nodes.find(node => node.id === connection.source), target = graph.nodes.find(node => node.id === connection.target);
  const output = port(catalog.find(spec => spec.type === source?.type)?.outputs, connection.sourceHandle);
  const input = port(catalog.find(spec => spec.type === target?.type)?.inputs, connection.targetHandle);
  if (!output || !input) return '請連接有效的輸出與輸入接點。';
  if (!types(input.accepts || input.type).some(type => ['any', '*'].includes(type) || types(output.type).includes(type))) return `接點類型不同：${output.type} 無法輸入 ${input.type}。`;
  if (graph.edges.some(edge => edge.source === connection.source && edge.target === connection.target && edge.sourceHandle === connection.sourceHandle && edge.targetHandle === connection.targetHandle)) return '這條連線已經存在。';
  if (!input.multiple && graph.edges.some(edge => edge.target === connection.target && edge.targetHandle === connection.targetHandle)) return '此輸入只能連接一個來源；請先刪除原有連線。';
  const pending = [connection.target], visited = new Set();
  while (pending.length) {
    const id = pending.pop();
    if (id === connection.source) return '流程不能形成循環，請依資料處理順序連接。';
    if (visited.has(id)) continue;
    visited.add(id);
    pending.push(...graph.edges.filter(edge => edge.source === id).map(edge => edge.target));
  }
  return '';
}

export function workflowGraphErrors(graph, catalog) {
  const errors = [];
  if (!graph.nodes.length) errors.push('請先載入模板，或新增至少一個節點。');
  for (const node of graph.nodes) {
    const spec = catalog.find(item => item.type === node.type);
    if (!spec) { errors.push(`不支援的節點：${node.type}`); continue; }
    for (const input of spec.inputs || []) if (input.required && !graph.edges.some(edge => edge.target === node.id && (edge.targetHandle === input.name || !edge.targetHandle && spec.inputs.length === 1))) errors.push(`${spec.label} 的「${input.label || input.name}」尚未連接。`);
  }
  for (const edge of graph.edges) {
    const error = workflowConnectionError(edge, { ...graph, edges: graph.edges.filter(item => item.id !== edge.id) }, catalog);
    if (error) errors.push(error);
  }
  return [...new Set(errors)];
}

export function normalizedPoint(clientX, clientY, rect) {
  if (!rect.width || !rect.height) return null;
  return [Math.max(0, Math.min(1, (clientX - rect.left) / rect.width)), Math.max(0, Math.min(1, (clientY - rect.top) / rect.height))].map(value => Math.round(value * 10000) / 10000);
}

export function polygonError(points) {
  if (!Array.isArray(points) || points.length < 3) return '區域至少需要 3 個頂點。';
  if (points.some(point => !Array.isArray(point) || point.length !== 2 || point.some(value => !Number.isFinite(value) || value < 0 || value > 1))) return '頂點必須是 0 到 1 之間的 [x, y] 座標。';
  const area = Math.abs(points.reduce((sum, point, index) => { const next = points[(index + 1) % points.length]; return sum + point[0] * next[1] - next[0] * point[1]; }, 0)) / 2;
  return area < 0.0001 ? '區域面積過小，請選擇至少 3 個不共線的頂點。' : '';
}

export function workflowRemoteNode(spec) {
  return Boolean(spec?.remote || spec?.cloud || /cloud|remote/i.test(spec?.category || '') || spec?.type === 'vlm');
}

export const workflowParameterValue = value => safeParameters(value);

export const workflowErrorText = value => typeof value === 'string' ? value : value?.message || value?.error || JSON.stringify(value);
export function workflowNodeStats(job) {
  const stats = job?.result?.node_stats || job?.node_stats || job?.progress_detail?.node_stats || {};
  return Array.isArray(stats) ? Object.fromEntries(stats.map(row => [row.node_id || row.id, row])) : stats;
}

export function workflowDraftNodeStats(job, runContext, requestKey) {
  // History may reuse node IDs for entirely different model/parameter graphs.
  return job && runContext?.jobId === job.id && runContext.requestKey === requestKey ? workflowNodeStats(job) : {};
}

export function workflowRemoteModels(integration) {
  if (!integration?.available || !Array.isArray(integration.models)) return [];
  return [...new Set(integration.models.filter(value => typeof value === 'string' && value.trim()))];
}
