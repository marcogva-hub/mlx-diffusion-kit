# TeaCache Calibration Coefficients

## Format

Each JSON file contains calibrated coefficients for a specific model.

### Fields

- `model` (string): Name of the model.
- `rel_l1_thresh` (float): Relative L1 distance threshold for step skipping.
- `poly_coeffs` (list[float]): Polynomial rescaling coefficients `[a0, a1, ..., an]`.
  Applied as: `calibrated = a0 + a1*x + a2*x^2 + ... + an*x^n`
  where `x` is the raw relative L1 distance.
- `notes` (string): Calibration context (number of steps, scheduler type, etc.).

### Example

```json
{
    "model": "CogVideoX",
    "rel_l1_thresh": 0.3,
    "poly_coeffs": [8.092438, -6.504696, 1.637915, -0.09498, 0.001356],
    "notes": "From TeaCache paper, CogVideoX-5B, 50 steps DDIM"
}
```

### Adding new coefficients

Use `scripts/calibrate_teacache.py` to generate coefficients for new models.
The calibration process:

1. Run the model on a reference dataset with all steps computed.
2. Record the relative L1 distances between consecutive modulated inputs.
3. Fit a polynomial to correct the distance bias.
4. Save the threshold and polynomial to a JSON file here.

### Available models

| File | Model | Steps | Scheduler |
|------|-------|-------|-----------|
| `cogvideox.json` | CogVideoX-5B | 50 | DDIM |
