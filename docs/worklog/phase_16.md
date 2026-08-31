# Phase 16 — Live Demo

One robot, streamed, the trained head in the loop. Watchable from a WebRTC client.

```bash
LIVESTREAM=1 PUBLIC_IP=<tailscale-ip> PYTHONEXE=$PWD/.venv-lerobot/bin/python \
  ~/isaacsim/python.sh src/scripts/eval_critic.py --model 20k --enable_cameras
```

The head was trained against 20k's features, so `--model 30k` and `--model 40k` ask whether
it survives a different policy's feature space. The script warns when they disagree.

| policy | succeeded |
|---|---|
| 20k | 15/28 |
| 30k | 3/8 |
| 40k | 10/16 |

Not separable at this size. 30k's success rate had never been measured.

Nothing is recorded — `eval_policy.py` owns that. So this measures the policy, not the
head. The clean same-policy number still needs a scored batch: `failure_score` per frame
written alongside the sim's own label.
