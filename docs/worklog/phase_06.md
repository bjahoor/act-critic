# Phase 06 — Train ACT

Stock ACT, 100k steps, on `bjahoor/lift-cube-franka`. HF Jobs, `rtx-pro-6000`, ~1h30m.

## 1. Setup

```bash
sudo pip install -U --break-system-packages huggingface_hub
hf buckets create act-runs
```

Checkpoints inside the job die with the container, and lerobot pushes to the Hub only after the final step.

## 2. Run

```bash
hf jobs run -d --flavor rtx-pro-6000 --timeout 12h --name act-lift-cube \
  -s HF_TOKEN \
  -e ACCELERATE_MIXED_PRECISION=bf16 \
  -v hf://buckets/bjahoor/act-runs:/out \
  ghcr.io/astral-sh/uv:python3.11-bookworm \
  uvx --python 3.11 --from lerobot==0.4.4 lerobot-train \
    --dataset.repo_id=bjahoor/lift-cube-franka \
    --dataset.video_backend=pyav \
    --policy.type=act \
    --policy.n_action_steps=10 \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=bjahoor/act-lift-cube-franka \
    --steps=100000 --batch_size=8 --num_workers=16 \
    --save_freq=25000 --log_freq=500 \
    --wandb.enable=false \
    --output_dir=/out/lift-cube-v5 --job_name=act_lift_cube
```

`n_action_steps=10` so chunks overlap — TCE needs it. `pyav` because the image has no ffmpeg and torchcodec will not
fall back. `lerobot==0.4.4` because the dataset is format v3.0.

`hf jobs logs <job_id>` · `hf jobs ps` · `hf jobs cancel <job_id>`

## 3. Publish checkpoints

Only the final model is pushed automatically. The 25k/50k/75k checkpoints sit in the bucket — pull each down and
upload it as its own repo. `training_state/` is optimizer state for resuming — skip it. Bucket-to-repo copy is unsupported, so it goes via disk.

```bash
for C in 025000:25k 050000:50k 075000:75k; do
  S=${C%%:*}; L=${C##*:}
  hf buckets sync hf://buckets/bjahoor/act-runs/lift-cube-v5/checkpoints/$S/pretrained_model ./ckpt-$L
  hf upload bjahoor/act-lift-cube-franka-$L ./ckpt-$L
done
```

## 4. Result

100k steps in 1h46m, $4.88. Final loss 0.098, grad norm 108 -> 5.0.

| checkpoint | model |
|---|---|
| 25k | https://huggingface.co/bjahoor/act-lift-cube-franka-25k |
| 50k | https://huggingface.co/bjahoor/act-lift-cube-franka-50k |
| 75k | https://huggingface.co/bjahoor/act-lift-cube-franka-75k |
| 100k | https://huggingface.co/bjahoor/act-lift-cube-franka |

Success rate: TBD.
