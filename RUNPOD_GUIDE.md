# Complete RunPod deployment guide

This guide deploys the step-200 `nagharsubject` high-noise LoRA from a Windows computer to a single-GPU RunPod Pod and generates silent Babaji image-to-video snippets.

## What goes where

Do not put every asset into one GitHub repository. Use this separation:

| Asset | Destination | Reason |
|---|---|---|
| `wan22_babaji_inference/` code | GitHub repository, then clone to `/workspace/wan22_babaji_inference` | Small, versioned code and documentation |
| 585 MB LoRA | `/workspace/nagharsubject_high_checkpoints/high/` | Too large for normal Git; identity-specific model asset |
| Four reference PNGs | `/workspace/prepared_wan22_i2v_curated/reference_images/` | Required I2V first frames; keep with private model assets |
| Wan base models | `/workspace/models/wan22_i2v_a14b/` | Automatically downloaded by the pipeline |
| Musubi Tuner | `/workspace/tools/musubi-tuner/` | Automatically installed by `setup_runpod.sh` |
| Python environment | `/workspace/venvs/wan22-babaji/` | Automatically installed by `setup_runpod.sh` |
| Generated videos | `/workspace/outputs/babaji_wan22_i2v/` | One timestamped directory per run |

Expected RunPod layout:

```text
/workspace/
├── wan22_babaji_inference/
├── nagharsubject_high_checkpoints/
│   └── high/
│       └── nagharsubject_high-step00000200.safetensors
├── prepared_wan22_i2v_curated/
│   └── reference_images/
│       ├── expressive_speaking.png
│       ├── neutral.png
│       ├── open_palm.png
│       └── raised_finger.png
├── models/wan22_i2v_a14b/
├── tools/musubi-tuner/
├── venvs/wan22-babaji/
└── outputs/babaji_wan22_i2v/
```

## Part 1: Prepare GitHub

### Why the LoRA should not go into normal GitHub history

The LoRA is 613,557,288 bytes (585.13 MiB). GitHub blocks files larger than 100 MiB in ordinary Git history. Git LFS can technically store a file this size, but direct SCP or a private Hugging Face model repository is simpler for a private identity LoRA and avoids consuming Git LFS storage/bandwidth.

The existing `.gitignore` excludes `.env`, models, caches, and outputs. The LoRA and source media are outside the inference project directory and will not be added by the commands below.

### Create an empty GitHub repository

1. Sign in to GitHub.
2. Select **New repository**.
3. Name it `wan22-babaji-inference`.
4. Choose **Private** unless you intentionally want the code public.
5. Do not initialize it with a README, `.gitignore`, or license—the local folder already contains the project.
6. Create the repository and copy its HTTPS URL.

### Push the local inference code from Windows PowerShell

Open PowerShell and run:

```powershell
Set-Location 'C:\Users\ab945\OneDrive\Documents\ChatGPT\data prep\wan22_babaji_inference'

git init -b main
git add .
git status
git commit -m "Add Wan 2.2 Babaji LoRA inference pipeline"
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/wan22-babaji-inference.git
git push -u origin main
```

Replace `YOUR_GITHUB_USERNAME`. Git for Windows normally opens Git Credential Manager or a browser for authentication. If `origin` already exists, inspect it with `git remote -v` and change it with:

```powershell
git remote set-url origin https://github.com/YOUR_GITHUB_USERNAME/wan22-babaji-inference.git
```

Before pushing, `git status` should list only project code and documentation. It must not list `.env`, `.safetensors`, source videos, or the prepared dataset.

### GitHub CLI alternative

If the GitHub CLI is installed:

```powershell
Set-Location 'C:\Users\ab945\OneDrive\Documents\ChatGPT\data prep\wan22_babaji_inference'
git init -b main
git add .
git commit -m "Add Wan 2.2 Babaji LoRA inference pipeline"
gh auth login
gh repo create YOUR_GITHUB_USERNAME/wan22-babaji-inference --private --source . --remote origin --push
```

## Part 2: Configure SSH for RunPod

Full SSH over a public IP is required for SCP. RunPod's basic proxy SSH connection does not support SCP/SFTP.

### Create a local SSH key if necessary

In Windows PowerShell:

```powershell
if (-not (Test-Path "$env:USERPROFILE\.ssh\id_ed25519")) {
    ssh-keygen -t ed25519 -f "$env:USERPROFILE\.ssh\id_ed25519" -C "runpod"
}
Get-Content "$env:USERPROFILE\.ssh\id_ed25519.pub"
```

Copy the complete public-key line. It begins with `ssh-ed25519`. Do not copy the fingerprint and never upload the private file `id_ed25519`.

In RunPod, open the account **Credentials** page, add the public key under SSH public keys, and save it.

## Part 3: Create storage and a GPU Pod

### Recommended first run

- GPU: one H100 80 GB or A100 80 GB.
- Template: official RunPod PyTorch template with Ubuntu and CUDA 12.8 compatibility.
- GPU count: 1.
- Deployment: On-Demand for the first setup and validation.
- Persistent storage: 150 GB.
- Container disk: 30–50 GB is sufficient because persistent assets live under `/workspace`.
- System RAM: 96–128 GB recommended.
- CPU: 16 vCPU recommended.
- SSH terminal access: enabled.
- Public IP / exposed TCP port 22: enabled if you will use SCP.

### Choose the persistence model

Recommended operational choice: create a **150 GB network volume** and attach it when deploying the Pod. It survives Pod deletion and can be attached to a replacement Pod. Network volumes are tied to a data center, so select a data center where your preferred 80 GB GPU is available.

RunPod currently makes network volumes available to Secure Cloud Pods. If you choose Community Cloud instead, use a sufficiently large volume disk and remember that it is tied to that Pod.

Privacy tradeoff: RunPod documents encryption-at-rest for volume disks, not network volumes. If encrypted-at-rest storage is mandatory for the identity media, choose an encrypted 150 GB volume disk instead, understand that it is deleted when the Pod is terminated, and back up the model and outputs before termination.

### Deploy using the console

1. Open RunPod and select **+ New → Pod**.
2. Select the latest official **RunPod PyTorch** template compatible with CUDA 12.8.
3. Enable **SSH terminal access** and provide the public key if prompted.
4. Select one **H100 80 GB** or **A100 80 GB**.
5. Select **On-Demand**.
6. Attach the 150 GB network volume or configure a 150 GB volume disk at `/workspace`.
7. Ensure a public IP and TCP port 22 are available for full SSH/SCP.
8. Review the hourly price and click **Deploy Pod**.
9. Wait until the Pod status is **Running**.

RunPod may show a Jupyter connection and two SSH choices. For file transfer, use the command labeled **SSH over exposed TCP**, which resembles:

```text
ssh root@194.26.196.6 -p 43201 -i ~/.ssh/id_ed25519
```

Record the displayed IP address and port. The examples below use `<POD_IP>` and `<POD_SSH_PORT>`.

## Part 4: Clone the code on RunPod

Open the Pod's web terminal or connect with the supplied SSH command.

### Public GitHub repository

```bash
cd /workspace
git clone https://github.com/YOUR_GITHUB_USERNAME/wan22-babaji-inference.git wan22_babaji_inference
```

### Private GitHub repository: read-only deploy key

On the Pod:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/github_deploy -N "" -C "runpod-deploy-key"
cat ~/.ssh/github_deploy.pub
```

Copy the public key. In the GitHub repository, open **Settings → Deploy keys → Add deploy key**, give it a name, paste the key, and leave write access disabled.

Then clone on the Pod:

```bash
cd /workspace
GIT_SSH_COMMAND='ssh -i ~/.ssh/github_deploy -o IdentitiesOnly=yes' \
  git clone git@github.com:YOUR_GITHUB_USERNAME/wan22-babaji-inference.git wan22_babaji_inference
```

Verify:

```bash
ls -la /workspace/wan22_babaji_inference
```

## Part 5: Upload the LoRA and reference images

### Create destination directories on the Pod

In the RunPod terminal:

```bash
mkdir -p /workspace/nagharsubject_high_checkpoints/high
mkdir -p /workspace/prepared_wan22_i2v_curated/reference_images
```

### Upload from Windows using SCP

Use the IP and external port from **SSH over exposed TCP**. In local Windows PowerShell:

```powershell
$PodIp = '<POD_IP>'
$PodPort = <POD_SSH_PORT>
$Key = "$env:USERPROFILE\.ssh\id_ed25519"
$Workspace = 'C:\Users\ab945\OneDrive\Documents\ChatGPT\data prep'

scp -P $PodPort -i $Key `
  "$Workspace\nagharsubject_high_checkpoints\high\nagharsubject_high-step00000200.safetensors" `
  "root@${PodIp}:/workspace/nagharsubject_high_checkpoints/high/"

scp -r -P $PodPort -i $Key `
  "$Workspace\prepared_wan22_i2v_curated\reference_images" `
  "root@${PodIp}:/workspace/prepared_wan22_i2v_curated/"
```

The LoRA is already compressed binary data, so compression rarely helps. If SCP is interrupted repeatedly, use `rsync` from WSL for resumable/incremental transfer or use a private Hugging Face repository as described below.

### Verify the uploads on RunPod

```bash
ls -lh /workspace/nagharsubject_high_checkpoints/high/
ls -lh /workspace/prepared_wan22_i2v_curated/reference_images/

sha256sum /workspace/nagharsubject_high_checkpoints/high/nagharsubject_high-step00000200.safetensors
```

The expected SHA-256 is:

```text
a05589023b0128de81a599e70632bec1a2e5cc7c172718182d25602f6e8311b4
```

The LoRA must be 613,557,288 bytes. If it is only a few hundred bytes, you copied a Git LFS/Xet pointer instead of the weight.

### Alternative: private Hugging Face model repository

This is useful if Pods will be recreated frequently.

On the local computer:

```powershell
python -m pip install --upgrade huggingface_hub
hf auth login
hf repos create YOUR_HF_USERNAME/nagharsubject-high-lora --private

hf upload YOUR_HF_USERNAME/nagharsubject-high-lora `
  'C:\Users\ab945\OneDrive\Documents\ChatGPT\data prep\nagharsubject_high_checkpoints\high\nagharsubject_high-step00000200.safetensors' `
  'nagharsubject_high-step00000200.safetensors'
```

On the Pod, after running the setup stage, authenticate using a read token supplied through a RunPod secret—not committed in `.env`—then download:

```bash
source /workspace/venvs/wan22-babaji/bin/activate
hf auth login
mkdir -p /workspace/nagharsubject_high_checkpoints/high
hf download YOUR_HF_USERNAME/nagharsubject-high-lora \
  nagharsubject_high-step00000200.safetensors \
  --local-dir /workspace/nagharsubject_high_checkpoints/high
```

Reference images can be uploaded to the same private repository or copied separately with SCP.

## Part 6: Configure and install the pipeline

On the Pod:

```bash
cd /workspace/wan22_babaji_inference
cp .env.example .env
nano .env
```

For the recommended 80 GB run, confirm these values:

```dotenv
RUNPOD_ROOT=/workspace
MODEL_ROOT=/workspace/models/wan22_i2v_a14b
OUTPUT_ROOT=/workspace/outputs/babaji_wan22_i2v
MUSUBI_DIR=/workspace/tools/musubi-tuner
VENV_DIR=/workspace/venvs/wan22-babaji
HF_HOME=/workspace/.cache/huggingface

LORA_PATH=/workspace/nagharsubject_high_checkpoints/high/nagharsubject_high-step00000200.safetensors
REFERENCE_ROOT=/workspace/prepared_wan22_i2v_curated/reference_images
REFERENCE_IMAGE=/workspace/prepared_wan22_i2v_curated/reference_images/open_palm.png

MEMORY_PROFILE=quality_offload
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LORA_STRENGTH=1.0
SEED=42
VERIFY_MODEL_HASHES=1
VERIFY_LORA_HASH=1
```

Do not add `HF_TOKEN` directly to a committed file. The public base-model downloads normally do not require it. If authentication is requested, configure `HF_TOKEN` as a RunPod secret/environment variable.

Install:

```bash
cd /workspace/wan22_babaji_inference
bash setup_runpod.sh 2>&1 | tee /workspace/setup_babaji_inference.log
```

The setup script installs system dependencies if needed, creates the persistent Python environment, installs a CUDA PyTorch build, and checks out the pinned Musubi Tuner revision.

Confirm the GPU and storage:

```bash
nvidia-smi
df -h /workspace
free -h
```

## Part 7: Inspect the LoRA and download base models

Inspect the uploaded checkpoint before spending time on the large downloads:

```bash
cd /workspace/wan22_babaji_inference
bash run_inference.sh inspect
```

Expected important fields:

```text
steps: 200
rank: 32
alpha: 32.0
base_model: wan2.2_i2v_high_noise_14B_fp16.safetensors
timestep_range: 900,1000
training_resolution: [768, 576]
training_frames: [81]
training_fps: 16.0
```

Download and hash-verify the approximately 69 GB base-model set:

```bash
bash run_inference.sh download 2>&1 | tee /workspace/download_wan22_models.log
```

This downloads:

- Wan 2.2 I2V-A14B low-noise FP16 DiT;
- Wan 2.2 I2V-A14B high-noise FP16 DiT;
- Wan 2.1 VAE used by the 14B model;
- UMT5 text encoder.

Both DiTs are required. Only the high-noise DiT receives this LoRA; the low-noise base expert performs the later denoising/detail steps.

The download command is restartable: rerun it after a network interruption. Existing complete files are reused and all model hashes are checked.

Run preflight:

```bash
bash run_inference.sh preflight
```

Then print the complete inference command without starting generation:

```bash
bash run_inference.sh generate --dry-run
```

The printed command must contain `--lora_weight_high_noise` and must not contain plain `--lora_weight`.

## Part 8: Generate the first video

Start with the default open-palm reference and prompt:

```bash
cd /workspace/wan22_babaji_inference
bash run_inference.sh generate 2>&1 | tee /workspace/first_generation.log
```

In another terminal, monitor:

```bash
watch -n 2 nvidia-smi
```

Output appears under a timestamped directory such as:

```text
/workspace/outputs/babaji_wan22_i2v/20261120_143000_single_seed42/
```

Each run contains the video plus `request.json`, `result.json`, and `inference.log`.

## Part 9: Prompt construction

### Required inputs

For each generation, supply:

1. A 768×576 first-frame PNG showing Babaji clearly.
2. A motion prompt. The pipeline automatically inserts the trigger `nagharsubject` if omitted.
3. A seed for reproducibility.
4. A LoRA strength, normally 0.8–1.0 for initial evaluation.

The base models, VAE, T5 encoder, high-noise LoRA mapping, dimensions, frame count, FPS, flow shift, and expert boundary are handled by the pipeline.

### Best reference-image practices

- Use the supplied 768×576 images initially; they match the training resolution and shot.
- Choose the reference that already resembles the requested motion:
  - `neutral.png` for calm address and subtle expressions;
  - `open_palm.png` for open-palm or blessing gestures;
  - `raised_finger.png` for one-finger emphasis;
  - `expressive_speaking.png` for stronger facial/speaking motion.
- Keep the subject's face sharp, unobstructed, and at the original scale.
- Keep clothing, background, crop, and camera angle close to the reference. Large scene/outfit changes weaken identity stability.
- Avoid an input with motion blur, text, watermark, another person, cropped face, or already malformed hands.
- Do not ask the model to move a visible hand into an anatomically incompatible starting pose.

### Prompt template

Use this order:

```text
nagharsubject, [appearance and fixed setting]. [one primary body/hand action].
[facial expression and subtle speaking motion]. [camera behavior]. [lighting and realism].
```

A strong prompt describes one achievable five-second motion. The image already supplies appearance, so spend most prompt detail on motion, expression, framing, and camera stability.

### Recommended default prompt

```text
nagharsubject, a middle-aged man wearing a cream knitted cap, a light beige kurta, and a cream shawl, seated outdoors in a pine forest in a static medium close-up. He slowly raises his right hand with an open palm in a calm blessing gesture, then holds it naturally while making subtle speaking mouth movements and gentle blinking. Locked camera, natural daylight, realistic anatomy and restrained motion.
```

Run it:

```bash
bash run_inference.sh generate \
  --reference /workspace/prepared_wan22_i2v_curated/reference_images/open_palm.png \
  --prompt "nagharsubject, a middle-aged man wearing a cream knitted cap, a light beige kurta, and a cream shawl, seated outdoors in a pine forest in a static medium close-up. He slowly raises his right hand with an open palm in a calm blessing gesture, then holds it naturally while making subtle speaking mouth movements and gentle blinking. Locked camera, natural daylight, realistic anatomy and restrained motion." \
  --seed 42 \
  --lora-strength 0.9
```

### Neutral address prompt

```text
nagharsubject, seated centered outdoors in the pine forest in a static medium close-up, looking calmly toward the camera. He makes subtle natural speaking mouth movements, a gentle blink, and a very small head nod while his hands remain relaxed. Locked camera, natural daylight, restrained realistic motion.
```

Use `neutral.png`.

### Raised-finger teaching prompt

```text
nagharsubject, seated outdoors in the pine forest in a static medium close-up. He gently holds one raised finger to emphasize a thoughtful point, with a warm composed expression, restrained head motion, and subtle natural speaking mouth movements. Locked camera, natural daylight, realistic hand anatomy.
```

Use `raised_finger.png`.

### Expressive teaching prompt

```text
nagharsubject, seated outdoors in the pine forest in a static medium close-up. He speaks expressively with a warm composed face and makes one small natural hand gesture near chest height, then returns to a relaxed pose. Locked camera, natural daylight, realistic motion and anatomy.
```

Use `expressive_speaking.png`.

### What to avoid in prompts

- Multiple sequential actions such as standing, turning, walking, pointing, and waving in one five-second clip.
- Exact spoken sentences; this generator produces no audio and cannot guarantee phoneme-level lip motion.
- Strong camera movement when the training shot used a locked camera.
- Sudden outfit/background transformations.
- Conflicting instructions such as `locked camera` and `camera circles around him`.
- Overly forceful motion words such as `rapidly`, `wildly`, or `dramatically` unless deliberately testing motion range.
- Repeating the trigger token many times.

The pipeline already supplies a negative prompt for blur, malformed hands/faces, extra fingers, extra people, camera motion, text, subtitles, and watermarks.

## Part 10: Systematic tuning for best results

Evaluate one variable at a time. Keep reference, prompt, resolution, steps, and seed constant.

### Establish a base-model control

Use a zero LoRA multiplier with the same prompt and seed:

```bash
bash run_inference.sh generate \
  --reference /workspace/prepared_wan22_i2v_curated/reference_images/open_palm.png \
  --prompt "YOUR EXACT PROMPT" \
  --seed 42 \
  --lora-strength 0.0
```

### Compare LoRA strengths

```bash
for strength in 0.70 0.85 1.00; do
  bash run_inference.sh generate \
    --reference /workspace/prepared_wan22_i2v_curated/reference_images/open_palm.png \
    --prompt "YOUR EXACT PROMPT" \
    --seed 42 \
    --lora-strength "$strength"
done
```

Interpretation:

- Weak learned motion or little difference from base: try 1.0.
- Repetitive/exaggerated movement or reduced prompt adherence: try 0.7–0.85.
- Identity/detail drift with healthy motion: this may reflect the absence of a low-noise identity LoRA; use the strongest suitable reference frame and test more seeds.

### Compare seeds

Once a strength is selected:

```bash
for seed in 42 43 44 45; do
  bash run_inference.sh generate \
    --reference /workspace/prepared_wan22_i2v_curated/reference_images/open_palm.png \
    --prompt "YOUR EXACT PROMPT" \
    --seed "$seed" \
    --lora-strength 0.85
done
```

### Guidance and steps

Start with 30 steps and guidance 5.0. Only after strength and seed testing:

- Guidance 4.5–5.5 is a sensible narrow test range.
- Higher guidance can increase prompt force but may reduce natural motion or introduce artifacts.
- More steps increase cost and do not guarantee improvement.
- Keep 768×576 and 81 frames for the first evaluation because they match training.

Example:

```bash
bash run_inference.sh generate \
  --prompt "YOUR EXACT PROMPT" \
  --reference /workspace/prepared_wan22_i2v_curated/reference_images/open_palm.png \
  --steps 30 \
  --guidance 4.5 \
  --seed 42 \
  --lora-strength 0.85
```

### Review every complete clip

Check:

- facial identity and detectability across all frames;
- eyes, teeth, lips, and cap/clothing stability;
- hand anatomy and finger count throughout the gesture;
- temporal flicker;
- whether motion matches the prompt;
- whether the camera remains locked;
- whether the LoRA improves the base control without collapsing motion diversity.

Do not select a checkpoint or seed from one favorable still frame.

## Part 11: Batch generation

The supplied JSONL contains four examples and loads the large models once:

```bash
cd /workspace/wan22_babaji_inference
bash run_inference.sh batch 2>&1 | tee /workspace/batch_generation.log
```

Edit it with:

```bash
nano /workspace/wan22_babaji_inference/prompts/babaji_snippets.jsonl
```

Each non-empty line is one JSON object:

```json
{"id":"clip_name","reference":"open_palm.png","seed":42,"prompt":"nagharsubject, one concise five-second action..."}
```

Relative reference names resolve under `REFERENCE_ROOT`. Keep each JSON object on one line.

## Part 12: Download outputs to Windows

List output directories on the Pod:

```bash
find /workspace/outputs/babaji_wan22_i2v -maxdepth 2 -type f -printf '%p\n'
```

Download the entire output tree using local PowerShell:

```powershell
$PodIp = '<POD_IP>'
$PodPort = <POD_SSH_PORT>
$Key = "$env:USERPROFILE\.ssh\id_ed25519"

scp -r -P $PodPort -i $Key `
  "root@${PodIp}:/workspace/outputs/babaji_wan22_i2v" `
  "$env:USERPROFILE\Downloads\"
```

Alternatively, use JupyterLab's file browser or RunPod Cloud Sync. Preserve `request.json`, `result.json`, and `inference.log` with each video.

## Part 13: Cost control and shutdown

1. Confirm videos and manifests exist under `/workspace`.
2. Download or back up important results.
3. Stop the Pod when not generating so GPU billing stops.
4. A volume disk survives a Pod stop but is deleted when that Pod is terminated.
5. A network volume survives Pod deletion and can be attached to another compatible Pod.
6. Container-disk content is disposable; do not store unique assets outside `/workspace`.

## Part 14: Troubleshooting

### `LoRA checkpoint not found`

```bash
grep LORA_PATH /workspace/wan22_babaji_inference/.env
ls -l /workspace/nagharsubject_high_checkpoints/high/
```

The filename and capitalization must match exactly.

### Checkpoint hash failure

Re-upload the 613,557,288-byte file and verify `sha256sum`. Do not disable LoRA verification to work around a genuinely different or truncated file.

### Base-model download receives 401/403

Create a Hugging Face read token, add it as a RunPod secret/environment variable named `HF_TOKEN`, restart the terminal/session if necessary, and rerun:

```bash
bash run_inference.sh download
```

Never paste the token into GitHub or commit it in `.env`.

### Out of disk space

```bash
df -h /workspace
du -sh /workspace/models /workspace/venvs /workspace/outputs 2>/dev/null
```

Increase persistent storage. Do not move the base models to the container disk as a long-term fix.

### CUDA unavailable

```bash
nvidia-smi
source /workspace/venvs/wan22-babaji/bin/activate
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Use an official CUDA-compatible PyTorch template and a GPU Pod rather than a CPU Pod.

### CUDA out of memory

First stop other GPU processes and check `nvidia-smi`. Then edit `.env`:

```dotenv
MEMORY_PROFILE=quality_offload
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

`quality_offload` preserves FP16 DiT weights but swaps eight transformer blocks through CPU memory, reducing the peak during model loading and LoRA merging. If that still fails, use:

```dotenv
MEMORY_PROFILE=balanced
```

The balanced profile targets 48 GB-class GPUs. On a 24 GB GPU use:

```dotenv
MEMORY_PROFILE=low_vram
```

Low-VRAM mode is much slower. Restart the generation command after changing `.env`.

### SCP connection failure

- Use **SSH over exposed TCP**, not RunPod's basic proxied SSH command.
- Verify the external port, IP, private-key path, and exposed TCP port 22.
- Ensure the Pod has a public IP.
- Confirm the full public key—not its fingerprint—was added to RunPod Credentials.

### Poor identity or hand quality

- Match the reference image to the requested gesture.
- Reduce motion complexity.
- Test seeds before increasing guidance.
- Compare strengths 0.7, 0.85, and 1.0.
- Keep the original 768×576 crop.
- Remember that the supplied adapter is high-noise-only; no trained low-noise identity/detail adapter is available.

### No audio

This is expected. The pipeline generates silent visual snippets. Add narration and lip-sync as a separate downstream stage after selecting a visually stable video.

## Official references

- RunPod: <https://docs.runpod.io/get-started>
- RunPod SSH: <https://docs.runpod.io/pods/configuration/use-ssh>
- RunPod file transfer: <https://docs.runpod.io/pods/storage/transfer-files>
- RunPod storage types: <https://docs.runpod.io/pods/storage/types>
- GitHub large files: <https://docs.github.com/en/repositories/working-with-files/managing-large-files>
- Hugging Face repositories: <https://huggingface.co/docs/hub/repositories-getting-started>
- Wan 2.2 I2V-A14B: <https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B>
- Musubi Wan documentation: <https://github.com/kohya-ss/musubi-tuner/blob/main/docs/wan.md>
