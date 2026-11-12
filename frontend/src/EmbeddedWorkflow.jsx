import React, { useEffect, useState } from 'react';
import { LoaderCircle, RefreshCw } from 'lucide-react';
import { api } from './api';
import LiveCamera from './LiveCamera';
import { snapshotLiveWorkflow } from './workflowLive';
import './EmbeddedWorkflow.css';

export function embeddedWorkflowId(search = globalThis.location?.search || '') {
  const value = new URLSearchParams(search).get('embedded_workflow') || '';
  return /^[a-f0-9]{32}$/.test(value) ? value : '';
}

export default function EmbeddedWorkflow({ workflowId, ...shared }) {
  const [workflow, setWorkflow] = useState(null), [error, setError] = useState(''), [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let alive = true;
    setError('');
    Promise.all([api(`/comfy/workflows/${workflowId}`), api('/workflows/catalog')]).then(([value, catalog]) => {
      if (!alive) return;
      const labels = value.graph.nodes.filter(node => ['detector', 'pose', 'skeleton_action', 'video_action', 'embedding', 'relation', 'hand_landmarks', 'prompt_mask'].includes(node.type)).map(node => {
        const label = catalog.node_types.find(spec => spec.type === node.type)?.label || node.type;
        return `${label}${node.params.model_id ? ` · ${node.params.model_id}` : ''}${node.params.checkpoint_id ? ' · 微調權重' : ''}`;
      });
      setWorkflow({ ...snapshotLiveWorkflow({ ...value, labels }), sourceMode: value.source_mode });
    }).catch(failure => { if (alive) setError(failure.message); });
    return () => { alive = false; };
  }, [workflowId, attempt]);
  if (error) return <div className="comfy-embedded-empty" role="alert"><h2>無法載入即時工作流</h2><p>{error}</p><button className="button secondary" onClick={() => setAttempt(value => value + 1)}><RefreshCw size={16}/>重試</button></div>;
  if (!workflow || !shared.system) return <div className="comfy-embedded-empty"><LoaderCircle size={25} className="spin"/><p>正在準備模型與即時影像來源</p></div>;
  return <div className="comfy-embedded-camera"><LiveCamera {...shared} workflow={workflow} initialSourceMode={workflow.sourceMode}/></div>;
}
