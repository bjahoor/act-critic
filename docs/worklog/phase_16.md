# Phase 16 — Live Demo

## 1. Running it

One robot, streamed, the trained head in the loop. Watchable from a WebRTC client.

```bash
LIVESTREAM=1 PUBLIC_IP=<lan-ip> PYTHONEXE=$PWD/.venv-lerobot/bin/python \
  ~/isaacsim/python.sh src/scripts/eval_critic.py --model 20k --enable_cameras
```

## 2. Three policies

The head was trained against 20k's features, so `--model 30k` and `--model 40k` ask whether
it survives a different policy's feature space. The script warns when they disagree.

| policy | succeeded |
|---|---|
| 20k | 15/28 |
| 30k | 3/8 |
| 40k | 10/16 |

Not separable at this size. 30k's success rate had never been measured.

## 3. What this does not measure

Nothing is recorded — `eval_policy.py` owns that. So this measures the policy, not the
head. The clean same-policy number still needs a scored batch: `failure_score` per frame
written alongside the sim's own label.

## 4. What the head looks at

`src/scripts/plot_attention.py` averages ABMIL's attention over a dataset and draws it as
two 7x7 camera grids. Run on the 10k set, 79,331 frames.

Token layout was verified rather than read off the source — perturbing one input at a time
and watching which encoder *inputs* move. Wrist changes 2-50, table 51-99, arm state 1 only,
so 0 is the latent. Blanking each corner of the table image confirmed row-major order.

| | |
|---|---|
| highest | 96, 97, 98 — 1.40%, 1.39%, 1.31%, adjacent, bottom of the table camera |
| lowest | 33, 34, 35 — 0.71%, 0.71%, 0.70%, adjacent, in the wrist camera |
| `z` | 0.97%, 65th of 100 |
| arm state | 0.80%, 82nd of 100 |
| top ten | 12.9%, where an average gives 10% |

The weighting is mild, and it favours the table camera over the wrist — the head watches the
scene, not the gripper. Whether that beats mean pooling is the ablation still to run.

![Attention weight of every token](../images/attention_grid.png)
