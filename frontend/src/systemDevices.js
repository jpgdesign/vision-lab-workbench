export function systemDeviceOptions(system) {
  const options = [{ value: 'auto', label: '自動選擇' }, { value: 'cpu', label: 'CPU' }];
  const seen = new Set(['auto', 'cpu']);
  for (const device of system?.devices || []) {
    if (!/^cuda:\d+$/.test(device.id) || seen.has(device.id)) continue;
    seen.add(device.id); options.push({ value: device.id, label: `${device.id} · ${device.name || 'NVIDIA GPU'}` });
  }
  return options;
}

export const deviceAvailable = (value, system) => !value || systemDeviceOptions(system).some(option => option.value === value);

export function applySystemDevices(models, system) {
  const options = systemDeviceOptions(system);
  const defaultValue = value => options.some(option => option.value === value) ? value : 'auto';
  const patch = parameter => parameter.key === 'device' ? { ...parameter, type: 'select', options, default: defaultValue(parameter.default) } : parameter;
  return models.map(model => {
    const capability = model.capabilities || {}, training = capability.training;
    const supportsTrainingDevice = training?.supported_params?.includes('device') || Object.hasOwn(training?.defaults || {}, 'device');
    const trainingParameters = (training?.parameters || []).map(patch);
    if (supportsTrainingDevice && !trainingParameters.some(parameter => parameter.key === 'device')) trainingParameters.push({ key: 'device', type: 'select', label: '運算裝置', options, default: defaultValue(training?.defaults?.device) });
    return { ...model, capabilities: { ...capability,
      inference_params: (capability.inference_params || []).map(patch),
      ...(training ? { training: { ...training, parameters: trainingParameters, ...(supportsTrainingDevice ? { defaults: { ...training.defaults, device: defaultValue(training.defaults?.device) } } : {}) } } : {}),
    } };
  });
}
