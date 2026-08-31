# Phase 08 — Failure Data

Roll out a fallible checkpoint and keep both outcomes.

## 1. Record

```bash
PYTHONEXE=$PWD/.venv-lerobot/bin/python ~/isaacsim/python.sh \
  src/scripts/eval_policy.py --model 20k --num_envs 12 --num_rollouts 200 --enable_cameras --headless \
  --record --dataset_root datasets/rollouts-20k-200 --repo_id bjahoor/lift-cube-rollouts-20k-200 --overwrite
```

12 envs, 25 minutes, peak 7.2 GB of 8. Headless saves ~450 MB over streaming.

## 2. Three sets

| Checkpoint | Episodes | Success | Frames success/fail | Use |
|---|---|---|---|---|
| 20k | 200 | 49% | 24/76 | Train |
| 10k | 200 | 25% | 9/91 | Transfer test |
| 20k | 100 | 58% | 32/68 | Fast offline loop |

The head trains per frame, and the ratio inverts there — failures run to the 500-step giveup, successes finish in ~150.

Balance decides whether the result is readable. A detector that says fail on every frame scores F1 0.86 on the 20k
set and 0.94 on the 10k one. The second is unusable: skill and skew are indistinguishable.

The transfer set is a different checkpoint, so it tests whether the head works on a policy it never saw. That only
holds if the trunks stay separate — do not pool.

## 3. One failure mode

Every failure across all three sets is a timeout. Zero drops. The drop check only fires when the cube leaves the table
entirely, and this policy fails by arriving misaligned and missing the grasp, then retrying until the giveup.

So the head is trained and evaluated on one failure mode. Any claim about generality is unsupported by this data.

## 4. Push

Chunks live outside the dataset directory, so they go up separately.

```bash
.venv-lerobot/bin/python src/scripts/push_dataset.py \
  --repo_id bjahoor/lift-cube-rollouts-20k-200 --root datasets/rollouts-20k-200
```

[train](https://huggingface.co/datasets/bjahoor/lift-cube-rollouts-20k-200) ·
[chunks](https://huggingface.co/datasets/bjahoor/lift-cube-rollouts-20k-200-chunks) ·
[transfer](https://huggingface.co/datasets/bjahoor/lift-cube-rollouts-10k) ·
[transfer chunks](https://huggingface.co/datasets/bjahoor/lift-cube-rollouts-10k-chunks)
