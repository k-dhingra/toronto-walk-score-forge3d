# Toronto Walk Score Forge3D

A controlled Forge3D render of Toronto's Walk Score surface, including fixed-sample checks for visible grain.

The script renders the same Toronto Walk Score surface three ways:

1. 64 fixed samples, no denoiser
2. 264 fixed samples, no denoiser
3. The 264-sample HDR result with A-Trous applied afterwards

The seed, camera, materials, lighting, and compositor stay fixed. Adaptive sampling is disabled so Forge3D cannot stop early.

## Run

Install Forge3D from its source checkout, then install the small Python stack used by the experiment:

```bash
pip install -e /path/to/forge3d
pip install numpy pillow
python experiment.py
```

Forge3D uses Vulkan here. Override `WGPU_BACKEND` before running if your build needs another backend.

Outputs are written to `output/`:

- `64-raw.png`
- `264-raw.png`
- `264-atrous.png`
- `comparison.png`
- `metrics.json`

## What this test can establish

If 64 and 264 raw samples look materially different, sampling noise is still a plausible cause. If they are nearly identical, sample count is unlikely to be the dominant cause.

The A-Trous pass is a separate ablation. It tests whether AOV-guided denoising removes the visible grain without smearing roads, boundaries, or colour transitions.

![Fixed-sample crop comparison](comparison.png)

The checked-in run used seed 31. The 64- and 264-sample raw crops differed by 0.037/255 mean absolute error, while their roughness values were effectively identical. See [`metrics.json`](metrics.json) for the raw output.

This experiment narrows the fault. It does not identify it by itself. Material texture, HDR/albedo division, and nearest-neighbour reprojection remain separate suspects.

## Input

`toronto_inputs.npz` contains the preprocessed heightfield, map texture, and Toronto mask used by the original render. It contains no addresses, URLs, or API credentials.
