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


def estimate_output_shape(
    latent_chunks: list[mx.array],
    spatial_upsample: int = 1,
    temporal_upsample: int = 1,
) -> tuple[int, ...]:
    """Estimate the decoded output shape for buffer pre-allocation.

    Assumes latent_chunks are [B, C, T, H, W] and decoded output has
    the same B, C but upsampled T, H, W.

    Args:
        latent_chunks: List of latent chunk tensors.
        spatial_upsample: Spatial upsampling factor (applied to H and W).
        temporal_upsample: Temporal upsampling factor (applied to T).

    Returns:
        Estimated output shape (B, C, T_total, H_out, W_out).
    """
    if not latent_chunks:
        raise ValueError("latent_chunks must be non-empty")

    ref = latent_chunks[0]
    B, C = ref.shape[0], ref.shape[1]
    total_t = sum(chunk.shape[2] for chunk in latent_chunks) * temporal_upsample
    H_out = ref.shape[3] * spatial_upsample
    W_out = ref.shape[4] * spatial_upsample
    return (B, C, total_t, H_out, W_out)


def preallocate_output_buffer(
    shape: tuple[int, ...],
    dtype: mx.Dtype = mx.float16,
) -> mx.array:
    """Create a pre-allocated zero buffer for streaming decode output.

    Args:
        shape: Output shape from estimate_output_shape.
        dtype: Data type for the buffer.

    Returns:
        Zero-initialized buffer of the specified shape and dtype.
    """
    return mx.zeros(shape, dtype=dtype)


def chunked_decode_with_cache(
    decode_fn: Callable[[mx.array, WaveletVAECache], tuple[mx.array, dict[int, mx.array]]],
    latent_chunks: list[mx.array],
    cache: WaveletVAECache,
    output_buffer: Optional[mx.array] = None,
    callback: Optional[Callable[[int, mx.array], None]] = None,
) -> mx.array:
    """Decode latent chunks sequentially, propagating causal conv states.

    Three modes:
      1. output_buffer provided: writes decoded chunks directly into the buffer
         (no intermediate list, constant memory).
      2. callback provided: calls callback(chunk_idx, decoded) after each chunk
         (for streaming/display). Still returns the full concatenated output.
      3. Neither: original behavior (accumulate list + concatenate).

    Args:
        decode_fn: Decoder wrapper with signature
            (latent_chunk, cache) -> (decoded_frames, new_conv_states).
        latent_chunks: List of latent tensors, one per temporal chunk.
        cache: WaveletVAECache instance.
        output_buffer: Optional pre-allocated buffer [B, C, T_total, H, W].
            When provided, decoded chunks are written in-place.
        callback: Optional function called with (chunk_idx, decoded_chunk)
            after each chunk is decoded.

    Returns:
        Concatenated decoded frames along temporal axis (dim=2).

    Raises:
        ValueError: If latent_chunks is empty.
    """
    if not latent_chunks:
        raise ValueError("latent_chunks must be non-empty")

    if output_buffer is not None:
        # Buffer mode: write directly into pre-allocated output
        t_offset = 0
        for i, chunk in enumerate(latent_chunks):
            decoded, new_states = decode_fn(chunk, cache)
            for idx, state in new_states.items():
                cache.set_state(idx, state)

            t_len = decoded.shape[2]
            output_buffer[:, :, t_offset:t_offset + t_len, :, :] = decoded
            t_offset += t_len

            if callback is not None:
                callback(i, decoded)

        return output_buffer

    # List accumulation mode (original + optional callback)
    frames = []
    for i, chunk in enumerate(latent_chunks):
        decoded, new_states = decode_fn(chunk, cache)
        for idx, state in new_states.items():
            cache.set_state(idx, state)
        frames.append(decoded)

        if callback is not None:
            callback(i, decoded)

    return mx.concatenate(frames, axis=2)
