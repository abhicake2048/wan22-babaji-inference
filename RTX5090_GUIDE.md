# RTX 5090 Linux inference and checkpoint-comparison guide

This guide deploys the Wan 2.2 I2V-A14B pipeline to a remote Linux computer with one NVIDIA GeForce RTX 5090, uploads the private LoRA and reference images separately from the code, generates persistent MP4 files, and compares the 200-, 400-, and 600-step LoRAs under identical conditions.

## Important: OpenRouter is separate

OpenRouter is an API router for hosted language models. It does not connect this pipeline to a local RTX 5090 and cannot run the Wan video model from these local LoRA files. Use SSH, a VPN, or the machine provider's terminal to access the 5090 system. OpenRouter may be used separately to draft prompts, but it is not required for video inference and no OpenRouter API key belongs in this repository.

## 1. Required remote hardware and software

Recommended configuration:

- NVIDIA GeForce RTX 5090 with 32 GB VRAM.
- Ubuntu 22.04 or 24.04, native Linux preferred.
- NVIDIA driver from the 570 series or newer and a CUDA 12.8-compatible PyTorch wheel.
- 96 GB system RAM recommended; 64 GB is the practical minimum. Block swapping can use a large amount of RAM.
- At least 120 GB free on fast local NVMe storage. The two DiTs, T5, VAE, Python environment, LoRAs, caches, and outputs must fit.
- PCIe 4.0/5.0 x16 is desirable because swapped transformer blocks move between RAM and GPU.
- `git`, Python 3.10 or newer, `ffmpeg`, and OpenSSH.

On the remote system, check the machine before uploading anything:

```bash
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
free -h
df -h
python3 --version
```

The GPU line must identify an RTX 5090 and roughly 32 GB memory. If `nvidia-smi` fails, or the driver is too old for the RTX 5090, ask the machine administrator to install a current NVIDIA driver before continuing.

## 2. What goes to GitHub and what stays private

Push only the small code and documentation in `wan22_babaji_inference` to a private GitHub repository. Do not commit:

- any `.safetensors` file;
- `.env` or API tokens;
- reference images if they contain a private person's likeness;
- generated videos.

The `.gitignore` supplied here excludes those paths. A LoRA is about 585 MiB, while ordinary GitHub repositories block files over 100 MiB. Transfer the LoRAs and reference images directly with SCP, SFTP, rsync, or private object storage.

## 3. Push the updated code to the existing GitHub repository

This folder is already a Git repository connected to `https://github.com/abhicake2048/wan22-babaji-inference.git`. Make sure that GitHub repository is private, then review and push the code from local PowerShell:

```powershell
Set-Location 'C:\Users\ab945\OneDrive\Documents\ChatGPT\data prep\wan22_babaji_inference'
git status --short
git add .
git diff --cached --stat
git commit -m "Add RTX 5090 inference and checkpoint comparison"
git push origin main
```

Before the commit, verify that the staged list contains only code, examples, tests, and documentation. It must not contain `.env`, `.safetensors`, private images, generated videos, or the `dist` bundle.

If you intentionally want a new repository instead, remove or change the existing remote only after checking it:

```powershell
git remote -v
```

## 4. Create the remote workspace and clone the code

Connect to the GPU computer:

```powershell
ssh -p <SSH_PORT> <REMOTE_USER>@<REMOTE_HOST>
```

The examples use `/data/wan22`. Change it everywhere if that disk does not exist:

```bash
export SYSTEM_ROOT=/data/wan22
sudo mkdir -p "$SYSTEM_ROOT"
sudo chown -R "$USER":"$(id -gn)" "$SYSTEM_ROOT"
mkdir -p "$SYSTEM_ROOT/nagharsubject_high_checkpoints/high" "$SYSTEM_ROOT/reference_images"
git clone <YOUR_PRIVATE_GITHUB_REPOSITORY_URL> "$SYSTEM_ROOT/wan22_babaji_inference"
cd "$SYSTEM_ROOT/wan22_babaji_inference"
cp .env.rtx5090.example .env
chmod +x setup_linux.sh run_inference.sh run_compare.sh
```

If the root is not `/data/wan22`, replace every path in `.env`:

```bash
nano .env
```

Keep `MEMORY_PROFILE=rtx5090_quality` while comparing checkpoints.

## 5. Prepare all LoRA files on the source computer

The remote comparison expects these exact names:

```text
nagharsubject_high-step00000200.safetensors
nagharsubject_high-step00000400.safetensors
nagharsubject_high-step00000600.safetensors
```

Only step 200 currently exists in this local workspace. Download/export the 400- and 600-step files from the training run before starting the three-way comparison. They must be high-noise checkpoints from the same training configuration; do not substitute a low-noise LoRA.

Generate a source-side SHA-256 manifest in PowerShell after all available checkpoints are in the folder:

```powershell
$CheckpointDir = 'C:\Users\ab945\OneDrive\Documents\ChatGPT\data prep\nagharsubject_high_checkpoints\high'
$Manifest = Join-Path $CheckpointDir 'checkpoints.sha256'
Get-ChildItem -LiteralPath $CheckpointDir -Filter '*.safetensors' | Sort-Object Name | ForEach-Object {
  $Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
  "$Hash  $($_.Name)"
} | Set-Content -LiteralPath $Manifest -Encoding ascii
Get-Content -LiteralPath $Manifest
```

The known step-200 digest is:

```text
a05589023b0128de81a599e70632bec1a2e5cc7c172718182d25602f6e8311b4
```

## 6. Upload LoRAs and reference images directly

In local PowerShell, set the connection details and upload each file. This example assumes the standard four curated reference images:

```powershell
$RemoteUser = '<REMOTE_USER>'
$RemoteHost = '<REMOTE_HOST>'
$SshPort = <SSH_PORT>
$Remote = "${RemoteUser}@${RemoteHost}"
$CheckpointDir = 'C:\Users\ab945\OneDrive\Documents\ChatGPT\data prep\nagharsubject_high_checkpoints\high'
$ReferenceDir = 'C:\Users\ab945\OneDrive\Documents\ChatGPT\data prep\prepared_wan22_i2v_curated\reference_images'

Get-ChildItem -LiteralPath $CheckpointDir -Filter '*.safetensors' | ForEach-Object {
  scp -P $SshPort -- $_.FullName "${Remote}:/data/wan22/nagharsubject_high_checkpoints/high/"
}
scp -P $SshPort -- (Join-Path $CheckpointDir 'checkpoints.sha256') "${Remote}:/data/wan22/nagharsubject_high_checkpoints/high/"
Get-ChildItem -LiteralPath $ReferenceDir -Filter '*.png' | ForEach-Object {
  scp -P $SshPort -- $_.FullName "${Remote}:/data/wan22/reference_images/"
}
```

For resumable transfer on an unstable connection, use `rsync --partial --progress` from WSL/Linux instead of SCP.

Verify every upload on the remote computer:

```bash
cd /data/wan22/nagharsubject_high_checkpoints/high
sha256sum -c checkpoints.sha256
ls -lh
```

Every listed checkpoint must report `OK`. Do not proceed with a failed hash.

## 7. Install the RTX 5090 runtime

Install basic OS packages if they are missing:

```bash
sudo apt-get update
sudo apt-get install -y git ffmpeg python3 python3-pip python3-venv
```

Then run the supplied installer:

```bash
cd /data/wan22/wan22_babaji_inference
bash setup_linux.sh
```

The installer creates an isolated virtual environment, installs a CUDA 12.8 PyTorch build, checks out the pinned Musubi commit, installs its dependencies, reports the GPU, and performs a CUDA matrix-multiplication smoke test. The final output must show CUDA available, RTX 5090, about 32 GiB, and a successful smoke test.

Do not install a CUDA 11.8, 12.1, 12.4, or 12.6 PyTorch wheel on this Blackwell GPU.

## 8. Download and validate the Wan base models

The base-model stage downloads approximately 69 GB and verifies pinned SHA-256 values:

```bash
cd /data/wan22/wan22_babaji_inference
bash run_inference.sh download
bash run_inference.sh inspect
bash run_inference.sh preflight
```

If Hugging Face requests authentication, export a read token in the shell or the machine's secret manager, then repeat the download:

```bash
export HF_TOKEN='hf_REPLACE_ME'
bash run_inference.sh download
unset HF_TOKEN
```

Never commit the token or put it into a public command history.

## 9. Run one test generation and save the video

Use a concise prompt with the `nagharsubject` trigger, one person, one gesture, a locked camera, and details that match the reference image:

```bash
cd /data/wan22/wan22_babaji_inference
bash run_inference.sh generate \
  --reference /data/wan22/reference_images/open_palm.png \
  --prompt "nagharsubject, a middle-aged man wearing a cream knitted cap, light beige kurta, and cream shawl, seated outdoors in a pine forest in a static medium close-up. He slowly raises his right hand with an open palm in a calm blessing gesture, with subtle natural speaking mouth movements. Locked camera, natural daylight, realistic face and hands." \
  --seed 42 \
  --lora-strength 1.0 \
  --height 576 --width 768 --frames 81 --fps 16 \
  --steps 30 --guidance 5.0
```

The script prints the timestamped output directory. The MP4, `request.json`, `result.json`, and `inference.log` are saved under:

```text
/data/wan22/outputs/babaji_wan22_i2v/
```

Confirm the file:

```bash
find /data/wan22/outputs/babaji_wan22_i2v -type f \( -name '*.mp4' -o -name 'result.json' \) -printf '%TY-%Tm-%Td %TH:%TM  %p\n' | sort
```

The generated clip is silent; narration and lip-sync are separate downstream stages.

## 10. Compare 200, 400, and 600 steps fairly

Run all checkpoints with the same prompt, input image, seed, LoRA strength, inference steps, and guidance:

```bash
cd /data/wan22/wan22_babaji_inference
bash run_compare.sh \
  --checkpoint-dir /data/wan22/nagharsubject_high_checkpoints/high \
  --checkpoint-steps 200 400 600 \
  --hash-manifest /data/wan22/nagharsubject_high_checkpoints/high/checkpoints.sha256 \
  --profile rtx5090_quality \
  --reference /data/wan22/reference_images/open_palm.png \
  --prompt "nagharsubject, a middle-aged man wearing a cream knitted cap, light beige kurta, and cream shawl, seated outdoors in a pine forest in a static medium close-up. He slowly raises his right hand with an open palm in a calm blessing gesture, with subtle natural speaking mouth movements. Locked camera, natural daylight, realistic face and hands." \
  --seed 42 --lora-strength 1.0 \
  --height 576 --width 768 --frames 81 --fps 16 \
  --infer-steps 30 --guidance 5.0
```

The runner continues through the list, saves each checkpoint under its own folder, and writes `comparison_summary.json` with checkpoint metadata, hashes, commands, durations, status, logs, and video paths. A missing 400/600 file is recorded as `missing` rather than silently replaced.

For a stronger decision, repeat the comparison with two more seeds and at least two reference/gesture combinations. Change only one variable per comparison. Review full videos for identity stability, temporal flicker, eyes/teeth, hand anatomy, clothing consistency, prompt following, and natural motion. A later training step is not automatically better; overtraining can reduce motion diversity or create artifacts.

## 11. Quality, speed, and OOM options on the 5090

Profiles:

- `rtx5090_quality`: scaled FP8, 12 swapped blocks, lazy expert loading. Use this to select the best checkpoint.
- `rtx5090_fast`: adds fast FP8 matrix arithmetic. It should be faster on RTX 40/50 series but Musubi warns of lower output quality.
- `rtx5090_safe`: swaps 20 blocks. Use it if the quality profile runs out of VRAM; expect slower inference.

Temporarily select a profile without editing `.env`:

```bash
MEMORY_PROFILE=rtx5090_fast bash run_inference.sh generate [the same generation options]
```

If CUDA runs out of memory:

```bash
nvidia-smi
MEMORY_PROFILE=rtx5090_safe bash run_inference.sh generate [the same generation options]
```

Stop other confirmed GPU workloads first. Do not choose `quality`, `quality_offload`, or `balanced` on a 32 GB 5090; their minimums target larger GPUs. Keep 576×768×81 for checkpoint selection because it matches training. Only after choosing a checkpoint should you test 20 inference steps or fewer frames for additional speed.

## 12. Download and preserve generated videos

From local PowerShell:

```powershell
$RemoteUser = '<REMOTE_USER>'
$RemoteHost = '<REMOTE_HOST>'
$SshPort = <SSH_PORT>
$Destination = "$env:USERPROFILE\Downloads\babaji_wan22_outputs"
New-Item -ItemType Directory -Force -Path $Destination | Out-Null
scp -r -P $SshPort "${RemoteUser}@${RemoteHost}:/data/wan22/outputs/babaji_wan22_i2v" $Destination
```

Preserve the JSON and log beside every MP4. They record which checkpoint, prompt, seed, profile, LoRA strength, and command produced the video.

## 13. Updating code later

On the local machine, commit and push changes. On the remote machine:

```bash
cd /data/wan22/wan22_babaji_inference
git pull --ff-only
bash setup_linux.sh
bash run_inference.sh preflight
```

The LoRAs, models, references, environment, and generated videos live outside the Git checkout and are not affected by `git pull`.

## Official references

- NVIDIA RTX 5090 specifications: <https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/>
- NVIDIA CUDA/Blackwell architecture matrix: <https://docs.nvidia.com/datacenter/tesla/drivers/latest/cuda-toolkit-driver-and-architecture-matrix.html>
- PyTorch installation selector: <https://pytorch.org/get-started/locally/>
- Musubi Wan documentation: <https://github.com/kohya-ss/musubi-tuner/blob/main/docs/wan.md>
- OpenRouter overview: <https://learn.openrouter.ai/courses/what-is-openrouter/what-is-openrouter>
- GitHub large-file limits: <https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github>
