export function branchPredictionGroups(result) {
  const explicit = result?.prediction_groups;
  const labels = { original: '原模型分支', new: '新增分支', other: '其他結果' };
  if (Array.isArray(explicit)) {
    const groups = ['original', 'new'].map(key => {
      const group = explicit.find(value => value?.branch === key && Array.isArray(value.predictions));
      return group ? { key, label: labels[key], predictions: group.predictions } : null;
    }).filter(Boolean);
    if (groups.length) return groups;
  }
  if (explicit && typeof explicit === 'object' && !Array.isArray(explicit)) {
    const keys = ['original', 'new'].filter(key => Array.isArray(explicit[key]));
    if (keys.length) return keys.map(key => ({ key, label: labels[key], predictions: explicit[key] }));
  }
  const predictions = Array.isArray(result?.predictions) ? result.predictions : [];
  if (!predictions.some(item => item?.branch === 'original' || item?.branch === 'new')) return [];
  return ['original', 'new', 'other'].map(key => ({ key, label: labels[key], predictions: predictions.filter(item =>
    key === 'other' ? !['original', 'new'].includes(item?.branch) : item?.branch === key) })).filter(group => group.predictions.length);
}
