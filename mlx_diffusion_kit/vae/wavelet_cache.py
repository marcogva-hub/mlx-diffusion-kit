"""B17 — WF-VAE Causal Cache: Conv state propagation for chunked VAE decode.

For chunked temporal decoding (SeedVR2, CogVideoX), causal 3D convolutions
recompute halo frames at chunk boundaries. This cache stores and propagates
the convolutional hidden states between chunks, eliminating redundant work.

Training-free, purely architectural optimization.
Estimated impact: -4 to -6% end-to-end on SeedVR2.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import mlx.core as mx


@dataclass
class WaveletCacheConfig:
    enabled: bool = True
    max_cached_layers: Optional[int] = None


class WaveletVAECache:
    """Cache for causal convolution states across temporal chunks.

    Stores per-layer hidden states that would otherwise be recomputed
    at chunk boundaries in chunked VAE decoding.
    """

    def __init__(self, config: Optional[WaveletCacheConfig] = None):
        self.config = config or WaveletCacheConfig()
        self._conv_states: dict[int, mx.array] = {}

    def get_state(self, layer_idx: int) -> Optional[mx.array]:
        """Retrieve cached conv state for a layer, or None if absent."""
        if not self.config.enabled:
            return None
        return self._conv_states.get(layer_idx)

    def set_state(self, layer_idx: int, state: mx.array) -> None:
        """Cache a conv state for a layer."""
        if not self.config.enabled:
            return
        if (
            self.config.max_cached_layers is not None
            and layer_idx not in self._conv_states
            and len(self._conv_states) >= self.config.max_cached_layers
        ):
            return
        self._conv_states[layer_idx] = state

    def clear(self) -> None:
        """Remove all cached states."""
        self._conv_states.clear()

    def num_cached(self) -> int:
        """Number of layers with cached states."""
        return len(self._conv_states)


def chunked_decode_with_cache(
    decode_fn: Callable[[mx.array, WaveletVAECache], tuple[mx.array, dict[int, mx.array]]],
    latent_chunks: list[mx.array],
    cache: WaveletVAECache,
) -> mx.array:
    """Decode latent chunks sequentially, propagating causal conv states.

    Args:
        decode_fn: Decoder wrapper with signature
            (latent_chunk, cache) -> (decoded_frames, new_conv_states).
            new_conv_states is a dict mapping layer_idx -> state tensor.
        latent_chunks: List of latent tensors, one per temporal chunk.
        cache: WaveletVAECache instance (may be pre-populated or empty).

    Returns:
        Concatenated decoded frames along temporal axis (dim=2).

    Raises:
        ValueError: If latent_chunks is empty.
    """
    if not latent_chunks:
        raise ValueError("latent_chunks must be non-empty")

    frames = []
    for chunk in latent_chunks:
        decoded, new_states = decode_fn(chunk, cache)
        for idx, state in new_states.items():
            cache.set_state(idx, state)
        frames.append(decoded)

    return mx.concatenate(frames, axis=2)
