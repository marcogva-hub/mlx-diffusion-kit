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

    Keyed by sha256(prompt). Uses mx.savez/mx.load for zero-copy MLX arrays.
    """

    def __init__(self, cache_dir: str | Path = "~/.cache/mlx-diffusion-kit/embeddings"):
        self._cache_dir = Path(cache_dir).expanduser()
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def _path(self, prompt: str) -> Path:
        return self._cache_dir / f"{self._key(prompt)}.npz"

    def get_or_compute(
        self, prompt: str, encoder_fn: Callable[..., mx.array], **kwargs
    ) -> mx.array:
        """Return cached embedding or compute, cache, and return."""
        path = self._path(prompt)
        if path.exists():
            return mx.load(str(path))["embeddings"]

        result = encoder_fn(prompt, **kwargs)
        mx.savez(str(path), embeddings=result)
        return result

    def clear(self) -> None:
        """Remove all cached embeddings."""
        for f in self._cache_dir.glob("*.npz"):
            f.unlink()

    def cache_size(self) -> int:
        """Number of cached entries."""
        return len(list(self._cache_dir.glob("*.npz")))
