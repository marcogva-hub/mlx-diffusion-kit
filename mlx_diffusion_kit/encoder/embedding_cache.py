"""B15 — Text Embedding Cache.

Pre-computes and caches text encoder outputs (T5, CLIP, etc.) on disk.
All 11 VSR models use fixed prompts, so the encoder only needs to run once.
"""

import hashlib
from pathlib import Path
from typing import Callable

import mlx.core as mx


class TextEmbeddingCache:
    """Disk-backed cache for text encoder embeddings.

    Keyed by sha256(encoder_id + prompt + kwargs). Uses mx.savez/mx.load
    for zero-copy MLX arrays. Writes are atomic (tmp + rename).
    """

    def __init__(self, cache_dir: str | Path = "~/.cache/mlx-diffusion-kit/embeddings"):
        self._cache_dir = Path(cache_dir).expanduser()
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, prompt: str, encoder_id: str = "default", **kwargs) -> str:
        key_str = f"{encoder_id}:{prompt}"
        for k in sorted(kwargs.keys()):
            key_str += f":{k}={kwargs[k]}"
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()

    def _path(self, prompt: str, encoder_id: str = "default", **kwargs) -> Path:
        return self._cache_dir / f"{self._key(prompt, encoder_id, **kwargs)}.npz"

    def get_or_compute(
        self,
        prompt: str,
        encoder_fn: Callable[..., mx.array],
        encoder_id: str = "default",
        **kwargs,
    ) -> mx.array:
        """Return cached embedding or compute, cache, and return.

        Args:
            prompt: Text prompt to encode.
            encoder_fn: Callable that takes (prompt, **kwargs) and returns mx.array.
            encoder_id: Identifier for the encoder (e.g., "t5-xxl", "clip-l").
                Different encoder_ids produce different cache keys for the same prompt.
            **kwargs: Additional arguments passed to encoder_fn (also included in cache key).
        """
        path = self._path(prompt, encoder_id, **kwargs)
        if path.exists():
            return mx.load(str(path))["embeddings"]

        result = encoder_fn(prompt, **kwargs)

        # Atomic write: write to tmp then rename
        # mx.savez appends .npz, so we use .tmp as stem suffix
        tmp_stem = path.with_suffix(".tmp")
        mx.savez(str(tmp_stem), embeddings=result)
        # mx.savez creates tmp_stem.npz
        tmp_actual = tmp_stem.with_suffix(".tmp.npz")
        tmp_actual.rename(path)

        return result

    def clear(self) -> None:
        """Remove all cached embeddings."""
        for f in self._cache_dir.glob("*.npz"):
            f.unlink()
        for f in self._cache_dir.glob("*.tmp"):
            f.unlink()

    def cache_size(self) -> int:
        """Number of cached entries."""
        return len(list(self._cache_dir.glob("*.npz")))
