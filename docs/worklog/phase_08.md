# Phase 08 — Failure Data

Roll out a fallible checkpoint and keep both outcomes. This is what the failure head trains on.

## 1. Record

```bash
LIVESTREAM=1 PUBLIC_IP=<server-ip> PYTHONEXE=$PWD/.venv-lerobot/bin/python ~/isaacsim/python.sh \
  scripts/eval_policy.py --model 20k --num_envs 8 --num_rollouts 100 --enable_cameras \
  --record --dataset_root datasets/rollouts-20k --repo_id bjahoor/lift-cube-rollouts-20k --overwrite
```

20k because it fails often enough to be interesting and succeeds often enough to be a policy. 8 envs, 8 minutes, peak
6.4 GB of 8.

100 episodes, 30721 frames.

| | Episodes | Frames |
|---|---|---|
| Success | 58 | 9760 |
| Failure | 42 | 20961 |

The episode ratio inverts at frame level: failures run to the 500-step giveup, successes finish in ~150. The head
trains per frame.

All 42 failures are timeouts, none dropped the cube. So a failure here is "reached, tried, never got there", and late
in such an episode the cube is visibly still on the table — a head could learn "late = failure" and beat the metric
while being useless. Earliness is the real result.

## 2. Push

Two repos. The chunks live outside the dataset directory, so they go up separately.

```bash
.venv-lerobot/bin/python scripts/push_dataset.py \
  --repo_id bjahoor/lift-cube-rollouts-20k --root datasets/rollouts-20k
```

[rollouts](https://huggingface.co/datasets/bjahoor/lift-cube-rollouts-20k) ·
[chunks](https://huggingface.co/datasets/bjahoor/lift-cube-rollouts-20k-chunks)
