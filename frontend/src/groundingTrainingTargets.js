const GROUNDING_MODEL = 'grounding_dino_base';

export function groundingTrainingTargets(modelId, checkpoint) {
  if (modelId !== GROUNDING_MODEL || !checkpoint?.training) return null;
  const training = checkpoint.training;
  const classes = (training.classes || []).filter(name => typeof name === 'string');
  const chunks = Array.isArray(training.caption_chunks) ? training.caption_chunks : [];
  const chunked = training.caption_mode === 'chunked' || chunks.length > 1;
  // Chunked checkpoints leave caption empty; caption_chunks holds every group.
  // Never substitute one group for the complete training vocabulary.
  const prompt = chunked ? '' : training.caption || (classes.length ? `${classes.join(' . ')} .` : '');
  if (!classes.length && !chunks.length && !prompt) return null;
  const classCount = classes.length || new Set(chunks.flatMap(chunk => chunk.global_class_ids || [])).size;
  return { classes, classCount, chunked, chunkCount: chunked ? chunks.length : 1, prompt };
}

export function selectInferenceCheckpoint(data, checkpointId, choices) {
  const targets = groundingTrainingTargets(data.modelId, choices.find(choice => choice.id === checkpointId));
  return {
    checkpointId, checkpointJobId: '', jobId: null, runMediaId: null,
    ...(data.modelId === GROUNDING_MODEL ? {
      params: { ...data.params, use_training_classes: Boolean(targets?.chunked) },
    } : {}),
  };
}

export function applyGroundingTrainingTargets(params, targets) {
  return targets?.chunked
    ? { ...params, use_training_classes: true }
    : { ...params, prompt: targets?.prompt || '', use_training_classes: false };
}

export function groundingParameterVisible(param, modelId, checkpoint, params) {
  if (param.key === 'use_training_classes') return false; // Render the contextual control once, beside its checkpoint.
  return !(['prompt', 'text_threshold'].includes(param.key) && params?.use_training_classes === true && groundingTrainingTargets(modelId, checkpoint));
}
