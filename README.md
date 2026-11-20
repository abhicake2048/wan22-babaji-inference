# Babaji Wan 2.2 I2V inference

This is a standalone inference harness for the downloaded step-200 `nagharsubject` high-noise LoRA. It generates silent ~5.1-second Babaji video snippets from a 768×576 first-frame image using Wan 2.2 I2V-A14B and Musubi Tuner.

The adapter is validated before every run and is attached only to the Wan 2.2 high-noise expert. The base low-noise expert remains unmodified because no low-noise LoRA is present in the supplied checkpoint folder.

## Files

```text
wan22_babaji_inference/
├── inference.py                  # inspect, download, preflight, single, batch
├── setup_runpod.sh              # pinned Musubi/PyTorch environment
├── run_inference.sh             # environment-aware launcher
├── .env.example                 # deployment paths and profile
├── requirements.txt
├── prompts/babaji_snippets.jsonl
├── tests/test_inference.py
├── RUNPOD_GUIDE.md
├── OOM_TROUBLESHOOTING.md
├── CHECKPOINT.md
├── HARDWARE.md
└── RESPONSIBLE_USE.md
```

For a complete first-time walkthrough—including GitHub upload, RunPod Pod creation, SCP commands for the LoRA and reference images, prompts, tuning, output download, and troubleshooting—follow [`RUNPOD_GUIDE.md`](RUNPOD_GUIDE.md).

## 1. Upload the workspace

Preserve this sibling layout under `/workspace`:

```text
/workspace/
├── wan22_babaji_inference/
├── nagharsubject_high_checkpoints/
│   └── high/nagharsubject_high-step00000200.safetensors
└── prepared_wan22_i2v_curated/
    └── reference_images/
        ├── neutral.png
        ├── open_palm.png
        ├── raised_finger.png
        └── expressive_speaking.png
```

On RunPod, allocate 150 GB of persistent storage and use an H100/A100 80 GB for the first fidelity evaluation. See `HARDWARE.md` before choosing a smaller GPU.

## 2. Configure and install

```bash
cd /workspace/wan22_babaji_inference
cp .env.example .env
bash setup_runpod.sh
```

The setup pins Musubi Tuner to the same known revision used by the training harness. It does not download the 69 GB base models yet.

## 3. Validate the actual LoRA

```bash
bash run_inference.sh inspect
```

This reads the safetensors metadata, confirms step 200, rank/alpha 32/32, timestep range 900–1000, the high-noise I2V base name, and the exact SHA-256 recorded in `CHECKPOINT.md`.

## 4. Download the pinned base models

```bash
bash run_inference.sh download
bash run_inference.sh preflight
```

The download stage retrieves both Wan 2.2 I2V-A14B FP16 experts, Wan 2.1 VAE, and UMT5 from pinned Hugging Face revisions, then verifies SHA-256 hashes. Wan 2.2 I2V-A14B does not use CLIP.

## 5. Generate one snippet

```bash
bash run_inference.sh generate
```

Custom example:

```bash
bash run_inference.sh generate \
  --reference /workspace/prepared_wan22_i2v_curated/reference_images/raised_finger.png \
  --prompt "A calm close-up teaching gesture with one raised finger and subtle speaking mouth motion" \
  --seed 1234 \
  --lora-strength 0.9
```

The `nagharsubject` trigger is added automatically if omitted. Begin with strength 0.8–1.0; if motion becomes repetitive or exaggerated, reduce the strength before changing other variables.

## 6. Generate the snippet batch

```bash
bash run_inference.sh batch
```

The JSONL examples cover neutral address, open-palm blessing, raised-finger emphasis, and expressive teaching. Batch mode resolves the image paths and gives Musubi one prompt file, so the large models are loaded once.

Each JSONL line accepts:

```json
{"id":"clip_name","reference":"open_palm.png","seed":42,"prompt":"nagharsubject, ...","negative_prompt":"optional override"}
```

Relative reference names resolve under `REFERENCE_ROOT`.

## Outputs and reproducibility

Each run creates a timestamped directory under `OUTPUT_ROOT` with:

- generated video file(s);
- `request.json` containing prompt, seed, profile, LoRA hash, and exact command;
- `inference.log` with Musubi output;
- `result.json` listing produced videos;
- `resolved_prompts.txt` for batch runs.

The default geometry matches the training data: 768×576, 81 frames, 16 FPS, 30 inference steps, flow shift 5, and expert boundary 900. Musubi's `--video_size` order is height then width, so the command passes `576 768`.

## Memory profiles

Edit `MEMORY_PROFILE` in `.env`:

- `quality_offload`: native FP16 plus eight CPU-swapped DiT blocks; recommended for 80 GB GPUs and preserves FP16 weight quality.
- `quality`: native FP16 with no block swapping; fastest, but the LoRA merge/cast peak can exceed an 80 GB GPU.
- `balanced`: scaled FP8 plus FP8 T5; intended for 48 GB-class GPUs.
- `low_vram`: FP8 plus 20-block CPU swapping; intended for 24 GB cards and considerably slower.

Dry-run any command before spending GPU time:

```bash
bash run_inference.sh generate --dry-run
```

## Tests

Tests do not need a GPU or base-model downloads:

```bash
python -m unittest discover -s tests -v
```

## Known limitations

- Only a high-noise LoRA was supplied. It primarily affects early motion and composition, not late identity/detail refinement.
- Step 200 is the first saved checkpoint, not proof of best quality. Compare against base Wan and any later checkpoint you train.
- I2V prompt control does not guarantee an exact hand trajectory.
- The generator creates silent video. Speech audio and lip-sync belong in a separate downstream pipeline.
- A recognizable-person model requires consent, disclosure, and human review; see `RESPONSIBLE_USE.md`.

Primary references: [Wan 2.2 I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B), [Musubi Wan documentation](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/wan.md), and [RunPod storage documentation](https://docs.runpod.io/pods/storage/types).

