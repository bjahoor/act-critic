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
    --save_freq=25000 --log_freq=500 \
    --wandb.enable=false \
    --output_dir=/out/v2-run1 --job_name=act_lift_cube_v2
```

`n_action_steps=10` so chunks overlap — TCE needs it. `pyav` because the image has no ffmpeg and torchcodec will not
fall back. `lerobot==0.4.4` because the dataset is format v3.0. 16 workers, not 32 — 32 exhausts the container's
shared memory and the dataloader dies at startup.

`hf jobs logs <job_id>` · `hf jobs ps` · `hf jobs cancel <job_id>`

## 3. Publish checkpoints

Only the final model is pushed automatically. The 25k checkpoint sits in the bucket — pull it down and upload it as
its own repo. `training_state/` is optimizer state for resuming — skip it. Bucket-to-repo copy is unsupported, so it
goes via disk.

```bash
hf buckets sync hf://buckets/bjahoor/act-runs/v2-run1/checkpoints/025000/pretrained_model ./ckpt-25k
hf upload bjahoor/act-lift-cube-franka-v2-25k ./ckpt-25k
```

## 4. Result

Discarded, see below. Rollout success at 100k was 16% over 25 rollouts; 25k and 50k were 5 rollouts each and too few
to rank.

## 5. Discarded run

A first run trained 100k steps on `bjahoor/lift-cube-franka` and produced four checkpoints. All discarded — two bugs
in the demo data, neither in training:

- `observation.state` carried the two finger positions, so the policy learned to copy its own gripper state instead
  of deciding. Across a 500-step rollout the predicted finger target never left 0.0398-0.0404. It never closed.
- The goal pose was randomized per episode and never given to the policy, so the task was not learnable.

Rollout success was 16% at 100k. The run had never been rollout-tested until after all four checkpoints existed —
that is how both bugs survived training. Test one checkpoint in the env before spending on a full run.

## 6. Retrain

`bjahoor/lift-cube-franka-v2`, same recipe with `--steps=50000`. `num_workers=16`: 32 crashes at startup on this
flavour — 23 vCPU and `/dev/shm` too small for the workers' buffers — and the loader was never the bottleneck.

Publishes to `bjahoor/act-lift-cube-franka-v2` and `-v2-25k`. The v1 repos stay up.

Success rate: TBD.
