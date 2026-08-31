# Phase 13 — Training Run

First real run of the head. `--pooling abmil`, defaults otherwise.

## 1. Result

200 episodes in, 64,300 frames. Stopped at epoch 11 on patience; best was epoch 6.

| | head | baseline |
|---|---|---|
| episode AP | **0.9914** | TCE alone 0.8292 |
| frame AP | 0.9744 | frame index 0.9135 |
| | | always-fail 0.4200 |

Loss fell 0.355 -> 0.106 while episode AP plateaued after epoch 6. That gap is the head
starting to memorise, and early stopping is what kept the epoch-6 weights.

## 2. What this number is not

The tune set is what early stopping selected on, so 0.9914 is a selection score, not a
held-out one. Two clean measurements are still outstanding: `bjahoor/lift-cube-rollouts-10k`,
a checkpoint the head has never seen, and the live in-the-loop eval.

Earliness is also unmeasured, and it is the claim that matters. A head that only fires once
the cube is visibly still on the table at step 400 scores exactly this.

## 3. Cost

Cache build 200 episodes, ~3 min. Training, ~7 s an epoch. The GPU is idle most of that —
the head is 0.1M parameters and the data path is the constraint, not the 3060 Ti.

## 4. Publishing it

The head is 392 KB — small, but weights do not belong in the source tree.

```bash
.venv-lerobot/bin/hf upload bjahoor/act-critic-head checkpoints/critic-abmil/critic.pt critic.pt
```

`--head` and `--critic` take either a local `.pt` or a hub repo id, and default to the repo,
so a fresh clone runs with nothing downloaded by hand.
