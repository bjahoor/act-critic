# Phase 16 — Live Demo

First runs of `eval_critic.py`. One robot, streamed, the trained head in the loop.

```bash
LIVESTREAM=1 PUBLIC_IP=<tailscale-ip> PYTHONEXE=$PWD/.venv-lerobot/bin/python \
  ~/isaacsim/python.sh src/scripts/eval_critic.py --model 20k --enable_cameras
```

## 1. Ran

Three checkpoints, the same head throughout — it was trained against 20k's features, so
30k and 40k are asking whether it survives a different policy's feature space. The script
warns when the trunk and the head disagree.

| policy | succeeded |
|---|---|
| 20k | 15/28 |
| 30k | 3/8 |
| 40k | 10/16 |

Small samples; the gaps are not separable at this size. 30k's success rate had never been
measured before and is not obviously better than 20k's, which is worth knowing before
picking a checkpoint to ship.

## 2. What this did not measure

Nothing is recorded, by design — `eval_policy.py` owns recording. So these runs show the
system works end to end and give a success rate for the policy, but no number for the head.
The clean same-policy measurement still needs a scored batch: `failure_score` per frame
written alongside the sim's own label.

## 3. Two things worth knowing

Isaac ignores SIGTERM. The process keeps stepping and holds its VRAM. SIGINT is what shuts
it down, and it takes a few seconds. `python.sh` is a wrapper, so killing the PID it
reports leaves the real process orphaned and still running — kill the `venv-lerobot/bin/python`
PID, not its parent.

The panel logs four `cannot convert float NaN to integer` errors in its first two seconds,
from `LiveLinePlot` autoscaling before it has data. Cosmetic, and it stops on its own.
