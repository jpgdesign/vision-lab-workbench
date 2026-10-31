import { parseWorkflowGraph } from './workflowGraph.js';

export function snapshotLiveWorkflow({ graph, name, options, labels }) {
  return { graph: parseWorkflowGraph(graph), name, options: { ...options }, labels: [...(labels || [])] };
}

export function workflowLiveRequest(workflow, captureFps, durationSeconds) {
  return {
    graph: parseWorkflowGraph(workflow.graph), options: { ...workflow.options },
    capture_fps: captureFps, max_duration_seconds: durationSeconds,
  };
}
