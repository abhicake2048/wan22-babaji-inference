# Restarting RTX 5090 inference

Use this after the initial setup and base-model download have already completed. Closing the old terminal does not remove the virtual environment, downloaded models, LoRAs, or generated videos under `/data/wan22`.

## Update the remote checkout once

In the `root@AIServer` Linux/WSL terminal:

```bash
cd /data/wan22/wan22_babaji_inference
git pull --ff-only
chmod +x setup_linux.sh run_inference.sh run_compare.sh start_inference.sh
```

The new launcher accepts a checkpoint step, any local first-frame image, a prompt, and a designated output directory.

## Windows-to-WSL paths

| Windows path | Linux/WSL path |
|---|---|
| `C:\Users\Pc\Downloads\my_image.jpg` | `/mnt/c/Users/Pc/Downloads/my_image.jpg` |
| `D:\babaji_inputs\my_image.png` | `/mnt/d/babaji_inputs/my_image.png` |
| `D:\lip_sync\generated` | `/mnt/d/lip_sync/generated` |

Do not pass a `C:\...` or `D:\...` path to the Linux script. Use the corresponding `/mnt/c/...` or `/mnt/d/...` path.

## Start a new inference

Check the GPU and enter the repository:

```bash
nvidia-smi
cd /data/wan22/wan22_babaji_inference
```

Example using step 400, a custom image on `D:`, a custom prompt, and an automatic lip-sync output destination:

```bash
bash start_inference.sh \
  --checkpoint-step 400 \
  --image /mnt/d/babaji_inputs/my_image.jpg \
  --save-to /mnt/d/lip_sync/generated \
  --prompt "nagharsubject, a middle-aged spiritual teacher seated outdoors in a static medium close-up. He looks calmly toward the camera and makes subtle natural speaking mouth movements. Locked camera, realistic face, natural daylight." \
  --seed 42 \
  --lora-strength 1.0 \
  --steps 30 \
  --guidance 5.0
```

Change `--checkpoint-step` to `200`, `400`, or `600`. The launcher validates that the file exists, reads its SHA-256 from `checkpoints.sha256` when available, and passes the chosen checkpoint to the high-noise expert.

The image may be PNG, JPG/JPEG, WEBP, or BMP. For the closest match to training, use a clear 768×576 landscape image with one visible subject, the desired starting pose, stable lighting, and no text or watermark.

## Designated output behavior

`--save-to` can be any writable Linux or mounted Windows directory. The pipeline creates a timestamped run folder inside it, for example:

```text
D:\lip_sync\generated\20261125_143000_single_seed42\
```

That folder contains the video, `request.json`, `result.json`, and `inference.log`. The parent directory also receives:

```text
D:\lip_sync\generated\latest_run.json
```

View the latest saved video path from Linux:

```bash
cat /mnt/d/lip_sync/generated/latest_run.json
```

List all generated videos:

```bash
find /mnt/d/lip_sync/generated -type f -iname '*.mp4' -printf '%TY-%Tm-%Td %TH:%TM  %s bytes  %p\n' | sort
```

## Keep inference running after closing the terminal

Start a persistent terminal session:

```bash
apt-get install -y tmux
tmux new -s wan_inference
```

Run `start_inference.sh` inside tmux. Detach with `Ctrl+B`, then `D`. Reattach later with:

```bash
tmux attach -t wan_inference
```

## Dry-run before spending GPU time

Add `--dry-run` to validate all paths and print the Musubi command without generating:

```bash
bash start_inference.sh \
  --checkpoint-step 600 \
  --image /mnt/d/babaji_inputs/my_image.jpg \
  --save-to /mnt/d/lip_sync/generated \
  --prompt "nagharsubject, calm speaking motion, locked camera" \
  --dry-run
```

## If the runtime or models are unavailable

You normally do not repeat setup or model download. Only use these recovery commands when their files are actually missing:

```bash
test -x /data/wan22/venvs/wan22-babaji/bin/python || bash setup_linux.sh
bash run_inference.sh preflight
```

If preflight reports missing base models or a missing verification marker:

```bash
bash run_inference.sh download
bash run_inference.sh preflight
```

If the 5090 runs out of memory, rerun the same launcher command with:

```text
--profile rtx5090_safe
```
