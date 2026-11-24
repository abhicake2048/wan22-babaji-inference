# Checkpoint inventory

The pipeline is bound to the downloaded file:

```text
nagharsubject_high_checkpoints/high/nagharsubject_high-step00000200.safetensors
```

Verified properties read directly from its safetensors header:

| Property | Value |
|---|---:|
| File size | 613,557,288 bytes (585.13 MiB) |
| SHA-256 | `a05589023b0128de81a599e70632bec1a2e5cc7c172718182d25602f6e8311b4` |
| Tensor entries | 1,200 |
| Training step | 200 |
| Epoch | 2 |
| Rank / alpha | 32 / 32 |
| Base DiT | `wan2.2_i2v_high_noise_14B_fp16.safetensors` |
| Timestep range | 900–1000 |
| Dataset resolution | 768×576 |
| Clip length / FPS | 81 frames / 16 FPS |
| Trigger token | `nagharsubject` |

This is a high-noise-only LoRA. Wan 2.2 I2V-A14B has two denoising experts. The adapter belongs only on the early/high-noise expert, so the command uses:

```text
--lora_weight_high_noise ... --lora_multiplier_high_noise 1.0
```

It intentionally does not pass `--lora_weight`, which would attach the file to the low-noise expert. The unmodified low-noise base model supplies later-stage facial, clothing, and texture detail.

Step 200 is an early checkpoint. Evaluate it against the base model and later checkpoints if they become available. A high-noise LoRA can improve learned motion/composition but cannot by itself provide the identity/detail refinement of a separately trained low-noise LoRA.

The inference validator now accepts later files named with the standard `-step########.safetensors` suffix when the embedded `ss_steps` metadata matches that filename and the model metadata still identifies the Wan I2V high-noise expert. Step 200 retains its built-in pinned digest. For 400/600 comparisons, create a SHA-256 manifest before upload and pass it to `compare_checkpoints.py`; see `RTX5090_GUIDE.md`.

