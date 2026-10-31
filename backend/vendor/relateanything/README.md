# RelateAnything local ONNX component

Based on RelateAnything by Maëlic Neau: <https://github.com/Maelic/RelateAnything>.

The relation runtime, postprocessor and scoring code are adapted from commit
`fc626d0d58113efcc8b08252c168a7ad529a171f`. Exact source URLs and hashes are
recorded in `sources.json`. See `LICENSE`, `NOTICE` and
`THIRD_PARTY_NOTICES.md`: upstream code is **AGPL-3.0-only**, and the additional
attribution requirement in NOTICE must be displayed in an interactive UI:
**“Based on RelateAnything by Maëlic Neau”**, linked to the project above.
The pretrained ConvNeXt-Large model contains DINOv3 weights and follows the
separate DINOv3 license; the repository's AGPL does not replace that license.

## Reproducible bundle and execution

Download from <https://huggingface.co/maelic/relsgg-convnext-large> the files
`relateanything.onnx`, `relateanything.json`, `predicate_bank.npz`, and
`thresholds.json`. `calibration.json`, `config.json` and the model README are
also retained with the local download provenance. The released ONNX file is
977,613,793 bytes, SHA256
`3a064f93bf6bfc40c597de761b6ad54ad002a91611d65d5f6a34b4d419dfdb1f`.
The 469,180-byte predicate bank SHA256 is
`d7e863f1609f1784767f6db69318ac8c23febcc000a005394885988d2fc514fd`.
No PyTorch checkpoint or text encoder is needed for the 243 bank predicates.

Tested on Windows 11, Python 3.11, NumPy 2.2.6 and **onnxruntime==1.23.2**,
with `CPUExecutionProvider` only and two intra-op threads. Installing this CPU
wheel does not require replacing PyTorch/CUDA. See the project's optional
workflow requirements for installation. Actual device is reported as `cpu`;
passing a GPU device to this adapter is an error, not a silent fallback.

Some Anaconda installations bundle MSVC 14.27 despite having a newer system
runtime. ORT 1.21+ requires MSVC 14.40 or newer
([Microsoft discussion](https://github.com/microsoft/onnxruntime/discussions/23971)).
Call `backend.workflow_relations.prepare_windows_runtime()` at process startup
**before importing NumPy, PyTorch or other native numerical libraries**. This
only loads DLLs by their absolute Windows system paths; it changes no installed
files. Install the Microsoft Visual C++ 2015–2022 x64 Redistributable if missing.
Restart an already-running process: late preloading cannot replace an older
library which is already loaded. Other operating systems take a no-op path.

## Preserved model contract and intentional changes

* `OnnxRelationHead` is the official relation-only runtime. It preserves plain
  square resize to 448×448 (not letterbox), BGR to RGB, `/255`, pixel xyxy to
  normalized cxcywh and zero-padding to 32 boxes. The network receives pixels
  and boxes; detector class names are carried as metadata only.
* Scores preserve the official calibrated formula
  `sigmoid(0.5825 * (predicate_logit + pair_logit) - 1.8911)` for this checkpoint.
  Detector confidences affect ranking, not the displayed score or threshold.
  The user-specified threshold is the actual global threshold. Bank-specific
  thresholds do not silently override that control.
* One predicate per ordered pair is selected using the official decoder.
  Unknown predicate names fail clearly. The bank contains `holding`, `wearing`,
  `riding`, `next to`, and `on`; any requested subset can be selected without
  downloading a text encoder. Unrestricted new text predicates require a new
  compatible bank and are not claimed here.
* The adapter selects the top 32 regions by detector confidence, stable by
  original index. It exposes selection/dropped counts in `last_diagnostics`.
  Returned indices always reference the caller's original detection list.
  Existing track keys are preserved; untracked regions never acquire invented
  person identities. More than 32 inputs are not silently presented as fully
  evaluated.
* Imports are relative, detector-specific code is omitted, and the scoring
  path is NumPy-only. Importing this component does not initialize CUDA or
  require a different PyTorch/Transformers version.
* The upstream legacy NPZ uses object arrays. Only the exact verified official
  bank may enable NumPy pickle loading. Other banks must use safe Unicode
  arrays. Unknown object/pickle arrays are rejected.

Real CPU graph tests verify loading, finite calibrated outputs, source-index
mapping and predicate selection. These checks establish runtime compatibility;
they do not establish action accuracy, safety suitability or real-time speed.
On this workstation the first relation forward was approximately 1.77 seconds,
so the node should run at its own inference interval rather than camera FPS.
