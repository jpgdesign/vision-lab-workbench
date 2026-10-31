import React from 'react';
import { branchPredictionGroups } from './predictionBranches';

export default function BranchPredictions({ result }) {
  const groups = branchPredictionGroups(result);
  if (!groups.length) return null;
  return <div className="branch-predictions" aria-label="各模型分支的預測">
    {groups.map(group => <section key={group.key} className={`prediction-branch ${group.key}`}><h4>{group.label}<span>{group.predictions.length} 個結果</span></h4>{group.predictions.length ? <div className="prediction-list">{group.predictions.slice(0, 6).map((prediction, index) => <span key={index}>{prediction.label}<b>{Number.isFinite(prediction.score) ? `${(prediction.score * 100).toFixed(1)}%` : ''}</b></span>)}{group.predictions.length > 6 && <span>+{group.predictions.length - 6}</span>}</div> : <p>此分支沒有超過閾值的結果。</p>}</section>)}
    <p className="branch-score-note">各分支分數獨立計算，不能視為所有類別加總為 100% 的共同機率。</p>
  </div>;
}
