"""B14.1 — DPM-Solver-v3: High-order ODE solver for diffusion models.

Achieves equivalent quality in 10-15 steps vs 20-50 for DDIM/Euler.
Uses log-SNR uniform spacing and multistep Taylor corrections.

Reference: Zheng et al., "DPM-Solver-v3: Improved Diffusion ODE Solver
with Empirical Model Statistics" (NeurIPS 2023).

This module provides the mathematical solver. It does NOT replace a model's
scheduler directly — it provides optimal timesteps and stepping logic
to be integrated into the inference pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx


class NoiseSchedule:
    """Noise schedule abstraction for VP (variance preserving) diffusion.

    Provides alpha_bar(t), log-SNR λ(t), and the inverse mapping.
    """

    def __init__(
        self,
        schedule_type: str = "vp",
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        num_train_timesteps: int = 1000,
    ):
        self.schedule_type = schedule_type
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_train_timesteps = num_train_timesteps

        # Precompute linear beta schedule and cumulative alpha_bar
        betas = mx.linspace(beta_start, beta_end, num_train_timesteps)
        alphas = 1.0 - betas
        self._alpha_bar_vals = mx.cumprod(alphas)

    def alpha_bar(self, t: mx.array) -> mx.array:
        """Cumulative product of (1 - beta) up to timestep t.

        Args:
            t: Timestep(s) in [0, num_train_timesteps - 1], continuous.

        Returns:
            ᾱ(t) values, same shape as t.
        """
        # Continuous interpolation into discrete schedule
        t_clamp = mx.clip(t, 0, self.num_train_timesteps - 1)
        t_floor = mx.floor(t_clamp).astype(mx.int32)
        t_ceil = mx.minimum(t_floor + 1, self.num_train_timesteps - 1)
        frac = t_clamp - t_floor.astype(t_clamp.dtype)

        ab_floor = self._alpha_bar_vals[t_floor]
        ab_ceil = self._alpha_bar_vals[t_ceil]
        return (1.0 - frac) * ab_floor + frac * ab_ceil

    def log_snr(self, t: mx.array) -> mx.array:
        """Log signal-to-noise ratio: λ(t) = log(ᾱ(t) / (1 - ᾱ(t))).

        Args:
            t: Timestep(s).

        Returns:
            λ(t) values. Monotonically decreasing (high SNR at t=0, low at t=T).
        """
        ab = self.alpha_bar(t)
        ab = mx.clip(ab, 1e-8, 1.0 - 1e-8)
        return mx.log(ab / (1.0 - ab))

    def inverse_log_snr(self, lsnr: mx.array) -> mx.array:
        """Inverse of log_snr: find t such that λ(t) ≈ lsnr.

        Uses binary search over the discrete schedule.

        Args:
            lsnr: Target log-SNR value(s).

        Returns:
            Timestep(s) corresponding to the target log-SNR.
        """
        # Compute log-SNR for all discrete timesteps
        all_t = mx.arange(self.num_train_timesteps, dtype=mx.float32)
        all_lsnr = self.log_snr(all_t)  # Decreasing

        # For each target lsnr, find the nearest timestep
        # all_lsnr is decreasing, so we search for insertion point
        result = []
        lsnr_flat = lsnr.reshape(-1)
        all_lsnr_np = all_lsnr.tolist()

        for val in lsnr_flat.tolist():
            # Binary search in decreasing sequence
            lo, hi = 0, len(all_lsnr_np) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if all_lsnr_np[mid] > val:
                    lo = mid + 1
                else:
                    hi = mid
            # Interpolate between lo-1 and lo
            if lo == 0:
                result.append(0.0)
            elif lo >= len(all_lsnr_np):
                result.append(float(self.num_train_timesteps - 1))
            else:
                lsnr_hi = all_lsnr_np[lo - 1]
                lsnr_lo = all_lsnr_np[lo]
                if abs(lsnr_hi - lsnr_lo) < 1e-10:
                    result.append(float(lo))
                else:
                    frac = (val - lsnr_lo) / (lsnr_hi - lsnr_lo)
                    result.append(float(lo) - frac)

        return mx.array(result).reshape(lsnr.shape)


@dataclass
class DPMSolverV3Config:
    order: int = 3
    num_steps: int = 15
    schedule_type: str = "vp"
    predict_type: str = "epsilon"  # "epsilon", "v_prediction", "x_start"
    corrector_steps: int = 0
    denoise_final: bool = True
    enabled: bool = True


def compute_optimal_timesteps(
    noise_schedule: NoiseSchedule, num_steps: int
) -> mx.array:
    """Compute log-SNR-uniform timesteps.

    Instead of uniform spacing in t, spaces uniformly in λ(t).
    This concentrates steps where the signal changes fastest.

    Args:
        noise_schedule: The noise schedule to use.
        num_steps: Number of denoising steps.

    Returns:
        Array of num_steps + 1 timesteps (including t=0), decreasing.
    """
    t_max = float(noise_schedule.num_train_timesteps - 1)
    lsnr_max = noise_schedule.log_snr(mx.array([0.0])).item()
    lsnr_min = noise_schedule.log_snr(mx.array([t_max])).item()

    # Uniform spacing in log-SNR domain
    lsnr_targets = mx.linspace(lsnr_max, lsnr_min, num_steps + 1)

    # Map back to timestep domain
    timesteps = noise_schedule.inverse_log_snr(lsnr_targets)
    return timesteps


class DPMSolverV3:
    """DPM-Solver-v3: multistep ODE solver for diffusion sampling.

    Supports orders 1 (≈DDIM), 2, and 3 with model output history
    for higher-order Taylor corrections.
    """

    def __init__(
        self,
        noise_schedule: NoiseSchedule,
        config: Optional[DPMSolverV3Config] = None,
    ):
        self.ns = noise_schedule
        self.config = config or DPMSolverV3Config()
        self._timesteps = compute_optimal_timesteps(self.ns, self.config.num_steps)
        self._model_outputs: list[mx.array] = []

    def get_timesteps(self) -> mx.array:
        """Return the optimal timestep sequence (num_steps + 1 values, decreasing)."""
        return self._timesteps

    @property
    def model_output_history(self) -> list[mx.array]:
        """Buffer of recent model predictions for multistep."""
        return self._model_outputs

    def _convert_to_epsilon(
        self, model_output: mx.array, sample: mx.array, t: mx.array
    ) -> mx.array:
        """Convert model output to noise prediction (epsilon) format."""
        if self.config.predict_type == "epsilon":
            return model_output
        elif self.config.predict_type == "v_prediction":
            ab = self.ns.alpha_bar(t)
            # Reshape for broadcasting: ab needs to match sample dims
            while ab.ndim < sample.ndim:
                ab = mx.expand_dims(ab, axis=-1)
            alpha = mx.sqrt(ab)
            sigma = mx.sqrt(1.0 - ab)
            return alpha * model_output + sigma * sample
        elif self.config.predict_type == "x_start":
            ab = self.ns.alpha_bar(t)
            while ab.ndim < sample.ndim:
                ab = mx.expand_dims(ab, axis=-1)
            alpha = mx.sqrt(ab)
            sigma = mx.sqrt(1.0 - ab)
            return (sample - alpha * model_output) / (sigma + 1e-8)
        else:
            raise ValueError(f"Unknown predict_type: {self.config.predict_type}")

    def _first_order_step(
        self,
        eps: mx.array,
        t_cur: mx.array,
        t_next: mx.array,
        sample: mx.array,
    ) -> mx.array:
        """First-order step (equivalent to DDIM).

        x_{t-1} = (α_{t-1}/α_t) * x_t - σ_{t-1} * (e^{-h} - 1) * ε
        where h = λ_{t-1} - λ_t
        """
        lsnr_cur = self.ns.log_snr(t_cur)
        lsnr_next = self.ns.log_snr(t_next)
        h = lsnr_next - lsnr_cur  # Positive (moving toward lower noise)

        ab_cur = self.ns.alpha_bar(t_cur)
        ab_next = self.ns.alpha_bar(t_next)

        # Reshape for broadcasting
        while ab_cur.ndim < sample.ndim:
            ab_cur = mx.expand_dims(ab_cur, -1)
            ab_next = mx.expand_dims(ab_next, -1)
            h = mx.expand_dims(h, -1)

        alpha_cur = mx.sqrt(ab_cur)
        alpha_next = mx.sqrt(ab_next)
        sigma_next = mx.sqrt(1.0 - ab_next)

        x_next = (alpha_next / alpha_cur) * sample - sigma_next * (mx.exp(-h) - 1.0) * eps
        return x_next

    def _second_order_step(
        self,
        eps_list: list[mx.array],
        t_cur: mx.array,
        t_next: mx.array,
        sample: mx.array,
    ) -> mx.array:
        """Second-order multistep: first-order + D1 correction."""
        if len(eps_list) < 2:
            return self._first_order_step(eps_list[-1], t_cur, t_next, sample)

        eps_cur = eps_list[-1]
        eps_prev = eps_list[-2]

        lsnr_cur = self.ns.log_snr(t_cur)
        lsnr_next = self.ns.log_snr(t_next)
        h = lsnr_next - lsnr_cur

        # D1: first-order finite difference of epsilon predictions
        # Use the log-SNR spacing between the two predictions
        t_prev_step = self._timesteps[max(0, self._current_idx - 1)]
        lsnr_prev = self.ns.log_snr(t_prev_step)
        r = (lsnr_cur - lsnr_prev) / (lsnr_next - lsnr_cur + 1e-8)

        D1 = eps_cur - eps_prev

        ab_cur = self.ns.alpha_bar(t_cur)
        ab_next = self.ns.alpha_bar(t_next)

        while ab_cur.ndim < sample.ndim:
            ab_cur = mx.expand_dims(ab_cur, -1)
            ab_next = mx.expand_dims(ab_next, -1)
            h = mx.expand_dims(h, -1)
            r = mx.expand_dims(r, -1)

        alpha_cur = mx.sqrt(ab_cur)
        alpha_next = mx.sqrt(ab_next)
        sigma_next = mx.sqrt(1.0 - ab_next)

        # First-order base
        x_next = (alpha_next / alpha_cur) * sample - sigma_next * (mx.exp(-h) - 1.0) * eps_cur
        # Second-order correction
        x_next = x_next - 0.5 * sigma_next * (mx.exp(-h) - 1.0) * D1
        return x_next

    def _third_order_step(
        self,
        eps_list: list[mx.array],
        t_cur: mx.array,
        t_next: mx.array,
        sample: mx.array,
    ) -> mx.array:
        """Third-order multistep: second-order + D2 correction."""
        if len(eps_list) < 3:
            return self._second_order_step(eps_list, t_cur, t_next, sample)

        eps_cur = eps_list[-1]
        eps_prev = eps_list[-2]
        eps_prev2 = eps_list[-3]

        D1 = eps_cur - eps_prev
        D1_prev = eps_prev - eps_prev2
        D2 = D1 - D1_prev

        lsnr_cur = self.ns.log_snr(t_cur)
        lsnr_next = self.ns.log_snr(t_next)
        h = lsnr_next - lsnr_cur

        ab_cur = self.ns.alpha_bar(t_cur)
        ab_next = self.ns.alpha_bar(t_next)

        while ab_cur.ndim < sample.ndim:
            ab_cur = mx.expand_dims(ab_cur, -1)
            ab_next = mx.expand_dims(ab_next, -1)
            h = mx.expand_dims(h, -1)

        alpha_cur = mx.sqrt(ab_cur)
        alpha_next = mx.sqrt(ab_next)
        sigma_next = mx.sqrt(1.0 - ab_next)

        exp_neg_h = mx.exp(-h)

        # First-order base
        x_next = (alpha_next / alpha_cur) * sample - sigma_next * (exp_neg_h - 1.0) * eps_cur
        # Second-order correction
        x_next = x_next - 0.5 * sigma_next * (exp_neg_h - 1.0) * D1
        # Third-order correction
        x_next = x_next - (1.0 / 6.0) * sigma_next * (exp_neg_h - 1.0) * D2
        return x_next

    def step(
        self,
        model_output: mx.array,
        timestep_idx: int,
        sample: mx.array,
    ) -> mx.array:
        """Execute one step of the DPM-Solver.

        Args:
            model_output: Raw model prediction (ε, v, or x₀ depending on predict_type).
            timestep_idx: Index into the timestep sequence (0-based).
            sample: Current noisy sample x_t.

        Returns:
            Denoised sample x_{t-1}.
        """
        t_cur = self._timesteps[timestep_idx:timestep_idx + 1]
        t_next = self._timesteps[timestep_idx + 1:timestep_idx + 2]
        self._current_idx = timestep_idx

        # Convert to epsilon prediction
        eps = self._convert_to_epsilon(model_output, sample, t_cur)
        self._model_outputs.append(eps)

        # Keep only what we need for the current order
        max_history = self.config.order
        if len(self._model_outputs) > max_history:
            self._model_outputs = self._model_outputs[-max_history:]

        # Select step order based on available history
        effective_order = min(self.config.order, len(self._model_outputs))

        if effective_order == 1:
            return self._first_order_step(eps, t_cur, t_next, sample)
        elif effective_order == 2:
            return self._second_order_step(self._model_outputs, t_cur, t_next, sample)
        else:
            return self._third_order_step(self._model_outputs, t_cur, t_next, sample)

    def reset(self) -> None:
        """Clear model output history for a new sampling run."""
        self._model_outputs.clear()
