# CUDA OOM diagnosis and recovery

## What the observed error means

The failed command used the `quality` profile on an NVIDIA A100-SXM4-80GB. The exception occurred before diffusion step 1 while Musubi was loading the high-noise 14B DiT, merging the high-noise LoRA, and casting the resulting model to FP16:

```text
torch.OutOfMemoryError: Tried to allocate 50.00 MiB
total capacity: 79.25 GiB
free: 29.94 MiB
process memory in use: 79.21 GiB
PyTorch allocated: 77.86 GiB
PyTorch reserved but unallocated: 868.96 MiB
```

The prompt and image pixels were not the cause: failure happened during model construction before the first denoising forward pass. The `quality` profile already used lazy expert loading and CPU VAE caching, but it did not use FP8 or transformer block swapping. The temporary peak from the 14B FP16 model plus LoRA merge/cast and the already prepared conditioning state filled essentially all 80 GB.

The allocator reported some fragmentation, but only about 0.87 GiB was reserved and unallocated. `expandable_segments` is useful as a secondary safeguard, not the main remedy: the denoising pass would still need working memory after the model finished loading.

## Recommended fix for A100/H100 80 GB

Update the RunPod code, then select `quality_offload`. This keeps FP16 weights but swaps eight DiT blocks through system RAM.

```bash
cd /workspace/wan22_babaji_inference
git pull

sed -i 's/^MEMORY_PROFILE=.*/MEMORY_PROFILE=quality_offload/' .env

if grep -q '^PYTORCH_CUDA_ALLOC_CONF=' .env; then
  sed -i 's/^PYTORCH_CUDA_ALLOC_CONF=.*/PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True/' .env
else
  printf '\nPYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n' >> .env
fi
```

Confirm the configuration:

```bash
grep -E '^(MEMORY_PROFILE|PYTORCH_CUDA_ALLOC_CONF)=' .env
bash run_inference.sh generate --dry-run
```

The dry-run command should contain all of these:

```text
--blocks_to_swap 8
--lazy_loading
--vae_cache_cpu
```

It should not contain `--fp8` for this profile.

Before retrying, confirm that no stale process is using the GPU:

```bash
nvidia-smi
```

If an old failed Python process is still listed, identify it first:

```bash
ps -fp <PID>
```

Stop only the confirmed stale inference process:

```bash
kill <PID>
sleep 5
nvidia-smi
```

Do not use `kill -9` unless a normal `kill` fails and the PID is definitely the stale job.

Retry the same generation command. The failed timestamped directory can remain as a diagnostic record; a new run gets a new directory.

## Fallback 1: scaled FP8

If `quality_offload` still runs out of memory, edit:

```bash
sed -i 's/^MEMORY_PROFILE=.*/MEMORY_PROFILE=balanced/' .env
bash run_inference.sh generate --dry-run
```

The command should now contain:

```text
--fp8
--fp8_scaled
--fp8_t5
--lazy_loading
--vae_cache_cpu
```

This substantially reduces DiT and text-encoder memory. Musubi documents `fp8_scaled` as higher quality than plain FP8, though still below FP16. Rerun inference with the original prompt and seed so the result is comparable.

## Fallback 2: FP8 plus heavier block swapping

If scaled FP8 still fails, use:

```bash
sed -i 's/^MEMORY_PROFILE=.*/MEMORY_PROFILE=low_vram/' .env
bash run_inference.sh generate --dry-run
```

This adds plain FP8, FP8 T5, and 20 swapped blocks. It uses much less VRAM but is slower and has the largest potential quality tradeoff.

## RTX 5090 (32 GB)

Do not use the A100/H100 `quality` or `quality_offload` profiles on an RTX 5090. Start with:

```dotenv
MEMORY_PROFILE=rtx5090_quality
```

This uses scaled FP8, FP8 T5, 12 swapped blocks, lazy expert loading, CPU VAE cache, and the lower-memory Wan 2.1-style time embedding. If it still OOMs after other GPU jobs are stopped, use:

```bash
MEMORY_PROFILE=rtx5090_safe bash run_inference.sh generate [the same generation options]
```

The safe profile swaps 20 blocks and therefore trades throughput for more VRAM headroom. `rtx5090_fast` is a speed/quality option, not an OOM fallback. The 5090 also needs a CUDA 12.8-compatible PyTorch build; run `setup_linux.sh` and confirm its CUDA smoke test before diagnosing a kernel-compatibility failure as an OOM.

## Do not start by changing these

- LoRA strength: affects model behavior, not the size of the loaded base DiT.
- Prompt wording: text length changes little here and was not the failing phase.
- Seed: does not reduce model-loading memory.
- Number of inference steps: changes runtime, but the model must still load before step 1.
- `torch.cuda.empty_cache()` from another shell: cannot free live tensors owned by the inference process.

Reducing resolution or frames can lower activation memory during denoising, but it cannot fix this specific pre-step model-casting peak as directly as offload/FP8. Keep the trained 768×576×81 geometry for the first retry.

## Non-fatal messages in the log

- `Failed to import sageattention`: harmless because the command explicitly uses SDPA.
- Hugging Face `404` for one T5 config probe: the tokenizer fallback succeeded, so this was not the failure.
- `timestep_boundary 900 ... setting to 0.9`: expected unit conversion.
- Unsupported `768*576` warning: Musubi's official size list does not include the training geometry, but it accepted the divisible-by-eight dimensions. This warning did not cause the model-loading OOM. Consider a supported 832×480 output only as a separate quality experiment, not as the first memory fix.

## Why the custom input was not responsible for OOM

The custom JPEG and the yellow/red clothing prompt may affect identity fidelity because they differ from the cream-clothing training distribution. They do not materially change the 14B model's parameter memory. The traceback proves memory exhausted during `model.to(dit_weight_dtype)` before denoising began.
