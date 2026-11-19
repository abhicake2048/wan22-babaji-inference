# Hardware and runtime requirements

## Recommended deployment

Use a Linux CUDA machine. The safest fidelity-first configuration is one NVIDIA H100 80 GB or A100 80 GB, at least 16 vCPU, 96–128 GB system RAM, and 150 GB persistent SSD storage. Put models, the virtual environment, the LoRA, reference images, and outputs under `/workspace` on RunPod.

| Profile | Practical GPU tier | VRAM gate | Quality / speed notes |
|---|---|---:|---|
| `quality` | H100 80 GB, A100 80 GB, A100 80 GB PCIe | 70 GiB detected | FP16 DiTs, best fidelity; recommended baseline |
| `balanced` | L40S 48 GB, RTX 6000 Ada 48 GB, A6000 48 GB | 40 GiB detected | Scaled FP8 DiTs and FP8 T5; lower memory, small quality tradeoff |
| `low_vram` | RTX 4090 24 GB or similar | 22 GiB detected | FP8 plus 20 swapped blocks; much slower and most likely to need tuning |

The low-VRAM row is a supported attempt, not a throughput guarantee. Musubi documents 24 GB inference through FP8 and block swapping, while the official Wan I2V-A14B single-GPU path specifies at least 80 GB VRAM. Start with `quality` for checkpoint evaluation.

## Storage budget

| Item | Approximate space |
|---|---:|
| High-noise DiT | 28.6 GB |
| Low-noise DiT | 28.6 GB |
| UMT5 encoder | 11.4 GB |
| Wan 2.1 VAE | 0.5 GB |
| Supplied LoRA | 0.61 GB |
| Python, PyTorch, Musubi, caches | 15–25 GB |
| Output headroom | 20+ GB |
| Recommended persistent allocation | 150 GB |

Both DiTs are required even though only the high-noise one receives the LoRA. Wan 2.2 switches from the high-noise expert to the low-noise expert at boundary 900.

## CPU, RAM, and platform

- OS: Ubuntu 22.04 or 24.04 on x86-64.
- GPU: CUDA-capable NVIDIA GPU; Ampere or newer is preferred.
- Driver: new enough for the CUDA runtime selected by `TORCH_INDEX_URL` (the default wheel index is CUDA 12.8).
- CPU: 8 vCPU minimum; 16+ recommended.
- RAM: 64 GB minimum for the low-VRAM profile; 96–128 GB recommended for model loading and offload headroom.
- Disk: persistent SSD or RunPod network volume. Avoid keeping unique weights only on the container disk.
- Network: roughly 70 GB of base-model downloads on first setup.

## Expected output and runtime

The default job is 768×576, 81 frames, 16 FPS, and 30 denoising steps: about 5.06 seconds of silent video. Runtime varies heavily by GPU and memory profile. Batch mode loads the models once and is substantially more efficient than launching each prompt separately.

Audio was not part of this LoRA inference interface. Add narration/lip-sync only as a separately validated downstream stage.

