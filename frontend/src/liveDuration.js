const units = {
  seconds: { seconds: 1, label: '秒' },
  minutes: { seconds: 60, label: '分鐘' },
  hours: { seconds: 3600, label: '小時' },
};

export function parseLiveDuration(value, unit, mode = 'custom') {
  if (mode === 'unlimited') return { seconds: null, label: '無時限', error: '' };
  const text = String(value ?? '').trim();
  if (mode !== 'custom' || !Object.hasOwn(units, unit)) return { error: '請選擇有效的時間限制與時間單位。' };
  if (!/^\d+$/.test(text) || Number(text) < 1) return { error: '時間請輸入大於 0 的整數，不可留空或含有小數。' };
  const amount = Number(text), seconds = amount * units[unit].seconds;
  if (!Number.isSafeInteger(amount) || !Number.isSafeInteger(seconds)) return { error: '時間數值過大，無法精確換算為秒；請輸入較小的整數。' };
  return { seconds, label: `${amount} ${units[unit].label}`, error: '' };
}

export function formatLiveDuration(seconds) {
  if (seconds === null) return '無時限';
  if (!Number.isSafeInteger(seconds) || seconds < 1) return '未設定';
  if (seconds % 3600 === 0) return `${seconds / 3600} 小時`;
  if (seconds % 60 === 0) return `${seconds / 60} 分鐘`;
  return `${seconds} 秒`;
}
