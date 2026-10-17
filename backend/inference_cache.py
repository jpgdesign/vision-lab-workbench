"""Small process-local LRU for the serial inference worker.

The owner must serialize acquire/use/discard. No tensors are moved to CPU on
eviction: dropping their owners releases GPU memory without allocating a second
copy of a potentially large model in RAM.
"""
from collections import OrderedDict
from dataclasses import dataclass
import gc
import hashlib
import os
from pathlib import Path
import sys


METADATA_NAMES = {
    "config.json", "preprocessor_config.json", "processor_config.json",
    "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json",
    "added_tokens.json", "vocab.txt", "vocab.json", "merges.txt",
    "model.safetensors.index.json", "pytorch_model.bin.index.json",
    "workbench_training.json", "training_config.json", "model_config.json",
    "generation_config.json",
}


@dataclass(frozen=True)
class ModelCacheKey:
    identity: tuple
    fingerprint: tuple


def _path_identity(path):
    # Retain HF snapshot file symlinks: the parent contains its configuration.
    return os.path.normcase(str(path.parent.resolve() / path.name))


def _file_fingerprint(path, *, metadata=False):
    info = path.stat()
    version = (_path_identity(path), str(path.resolve()), info.st_size,
               info.st_mtime_ns, info.st_ctime_ns, info.st_ino)
    if metadata:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(65536), b""):
                digest.update(block)
        version += (digest.hexdigest(),)
    return version


def model_cache_key(model_id, device, source, *, checkpoint=False, folder_model=False,
                    extra_metadata=()):
    """Fingerprint only model inputs, never media or inference-time parameters.

    Large weights use filesystem identity/version, while config/tokenizer/class
    metadata also uses a content digest. HF shards are included together with
    their index. RF architecture options live in workbench_training.json.
    """
    source = Path(source)
    folder = source if source.is_dir() else source.parent
    files = {source} if source.is_file() else set()
    for path in folder.iterdir():
        if path.is_file() and (path.name in METADATA_NAMES or
                ((folder_model or source.is_dir()) and path.suffix.lower() in {".bin", ".safetensors", ".pth", ".pt", ".pyth"})):
            files.add(path)
    extras = {Path(path) for path in extra_metadata}
    files.update(extras)
    if not files:
        raise ValueError("找不到可建立模型快取識別的本機權重。")
    identity_source = folder if folder_model else source
    identity = (model_id, str(device), "checkpoint" if checkpoint else "pretrained", _path_identity(identity_source))
    versions = tuple(_file_fingerprint(path, metadata=path.name in METADATA_NAMES or path in extras)
                     for path in sorted(files, key=lambda item: str(item)))
    return ModelCacheKey(identity, versions)


def release_unused_memory():
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            # Cleanup must never replace the inference error being reported.
            pass


class InferenceModelCache:
    def __init__(self, max_entries=4):
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        self.max_entries = max_entries
        self._entries = OrderedDict()

    def __len__(self):
        return len(self._entries)

    def acquire(self, key, loader):
        if key in self._entries:
            self._entries.move_to_end(key)
            return self._entries[key], True
        # Invalidate the prior version immediately, before loading replacement.
        stale = [old for old in self._entries if old.identity == key.identity]
        for old in stale:
            del self._entries[old]
        evicted = bool(stale)
        while len(self._entries) >= self.max_entries:
            self._entries.popitem(last=False)
            evicted = True
        if evicted:
            release_unused_memory()
        value = loader()
        self._entries[key] = value
        return value, False

    def discard(self, key, *, release=True):
        self._entries.pop(key, None)
        if release:
            release_unused_memory()

    def clear(self, *, release=True):
        self._entries.clear()
        if release:
            release_unused_memory()

    def close(self):
        self.clear()
