"""mlx-diffusion-kit: Inference optimizations for diffusion models on MLX."""

from mlx_diffusion_kit.__version__ import __version__
from mlx_diffusion_kit.encoder.embedding_cache import TextEmbeddingCache

__all__ = ["__version__", "TextEmbeddingCache"]
