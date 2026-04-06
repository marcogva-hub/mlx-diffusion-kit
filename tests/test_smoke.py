"""Smoke test: verify package imports and version."""


def test_import():
    import mlx_diffusion_kit

    assert mlx_diffusion_kit.__version__ == "0.1.0"


def test_submodules_importable():
    import mlx_diffusion_kit.cache
    import mlx_diffusion_kit.tokens
    import mlx_diffusion_kit.gating
    import mlx_diffusion_kit.attention
    import mlx_diffusion_kit.quality
    import mlx_diffusion_kit.scheduler
    import mlx_diffusion_kit.encoder
    import mlx_diffusion_kit.vae
    from mlx_diffusion_kit.orchestrator import DiffusionOptimizer

    opt = DiffusionOptimizer()
    assert opt.is_single_step is False
