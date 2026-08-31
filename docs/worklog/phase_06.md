# Phase 06 — Train ACT

Stock ACT, 50k steps, on `bjahoor/lift-cube-franka-v2`. HF Jobs, `rtx-pro-6000` (23 vCPU, $2.75/hr), ~50m.

## 1. Setup

```bash
sudo pip install -U --break-system-packages huggingface_hub
hf buckets create act-runs
```

Checkpoints written inside the job die with the container, and lerobot pushes to the Hub only after the final step.

## 2. Run

```bash
hf jobs run -d --flavor rtx-pro-6000 --timeout 4h --name act-lift-cube-v2 \
  -s HF_TOKEN \
  -e ACCELERATE_MIXED_PRECISION=bf16 \
  -v hf://buckets/bjahoor/act-runs:/out \
  ghcr.io/astral-sh/uv:python3.11-bookworm \
  uvx --python 3.11 --from lerobot==0.4.4 lerobot-train \
    --dataset.repo_id=bjahoor/lift-cube-franka-v2 \
    --dataset.video_backend=pyav \
    --policy.type=act \
    --policy.n_action_steps=10 \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=bjahoor/act-lift-cube-franka-v2 \
    --steps=50000 --batch_size=8 --num_workers=16 \
    --save_freq=10000 --log_freq=500 \
    --wandb.enable=false \
    --output_dir=/out/v2-run2 --job_name=act_lift_cube_v2
```

`n_action_steps=10` so chunks overlap — TCE needs it. `pyav` because the image has no ffmpeg and torchcodec will not
fall back. `lerobot==0.4.4` because the dataset is format v3.0. 16 workers, not 32 — 32 exhausts the container's
shared memory and the dataloader dies at startup.

`hf jobs logs <job_id>` · `hf jobs ps` · `hf jobs cancel <job_id>`

## 3. Publish checkpoints

lerobot pushes only the final model, and only after the last step (`lerobot_train.py:534`, after the loop). The
10k-40k checkpoints are written to the bucket as they happen, so publish them from here instead of waiting:

```bash
for S in 010000 020000 030000 040000; do
  L=$((10#$S / 1000))k
  hf buckets sync hf://buckets/bjahoor/act-runs/v2-run2/checkpoints/$S/pretrained_model ./ckpt-$L
  hf upload bjahoor/act-lift-cube-franka-v2-$L ./ckpt-$L
done
```

Each checkpoint dir is a complete inference package. `training_state/` is optimizer state for resuming — skip it.
Bucket-to-repo copy is unsupported, so it goes via disk. Run this on a poll during training and each model is usable
about a minute after it is written, rather than an hour later.

## 4. Result

50k steps in 51m, $2.32. Final loss 0.078, grad norm 108 -> 5.8.

| checkpoint | model |
|---|---|
| 10k | https://huggingface.co/bjahoor/act-lift-cube-franka-v2-10k |
| 20k | https://huggingface.co/bjahoor/act-lift-cube-franka-v2-20k |
| 30k | https://huggingface.co/bjahoor/act-lift-cube-franka-v2-30k |
| 40k | https://huggingface.co/bjahoor/act-lift-cube-franka-v2-40k |
| 50k | https://huggingface.co/bjahoor/act-lift-cube-franka-v2 |

Success rate: TBD.

## 5. Discarded run

A first run trained 100k steps on `bjahoor/lift-cube-franka` and produced four checkpoints. All discarded — two bugs
in the demo data, neither in training:

- `observation.state` carried the two finger positions, so the policy learned to copy its own gripper state instead
  of deciding. Across a 500-step rollout the predicted finger target never left 0.0398-0.0404. It never closed.
- The goal pose was randomized per episode and never given to the policy, so the task was not learnable.

Rollout success was 16% at 100k over 25 rollouts; the 25k and 50k checkpoints got 5 rollouts each, too few to rank.
Nothing had been rollout-tested until after all four checkpoints existed — that is how both bugs survived training.
Test one checkpoint in the env before spending on a full run. The v1 repos stay up as a record; they take a 9-dim
`observation.state` and will not load against v2 data.
