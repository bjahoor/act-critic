# Phase 14 — Held-Out Measurement

`src/scripts/measure_critic.py`. The numbers; `eval_critic.py` is the live demo.

```bash
PYTHONPATH=$PWD/src .venv-lerobot/bin/python src/scripts/measure_critic.py --hold 25
```

## 1. Alarm rule

A score over the threshold for one frame is noise. An alarm fires only after the score
holds above it for 25 frames, 0.5 s. TCE goes through the identical rule, or the
comparison is not one.

Operating point is 0.95 held 0.5 s. No threshold lives in the model; the sweep is the
output and where to draw the line is a deployment choice.

## 2. Result

`bjahoor/lift-cube-rollouts-10k`, 200 episodes, 150 failures. Never trained on, never used
for early stopping.

| at 0.95, held 0.5 s | head | TCE alone |
|---|---|---|
| failures caught | 94% | 10% |
| successes interrupted | 4% | 0% |
| first alarm | 1.86 s | 2.22 s |

TCE cannot reach a low false-alarm rate and keep recall: at its own best threshold it
catches everything but interrupts 36% of successes. Episode AP is 0.9964 against 0.9856,
which flatters TCE — the set is 75% failures, so the floor is 0.75 and AP compresses the
gap the table above shows plainly.

The grasp is decided between 1.0 and 1.8 s, so 1.86 s is a warning at the end of the
decisive window and 8 s before the giveup.

## 3. It partly predicts rather than detects

At frame 0, before the arm has moved, failures score 0.513 and successes 0.318. With a
5-frame hold, 20% of failures alarm within 0.5 s — before anything has gone wrong.

So two mechanisms are mixed: a prior on which starting cube positions this policy tends to
fail from, and watching the approach. The first is not cheating — layout genuinely predicts
difficulty — but it is a weaker claim than detecting a failure as it happens, and it is the
part most likely to break on a different scene distribution.

The 0.5 s hold suppresses it: nothing survives half a second above 0.95 that early.

## 4. Not yet run

The head has vision, TCE and ACM. Beating TCE alone does not show the vision earned its
place. Three ablations outstanding, all sharing the cache:

- TCE + ACM trained, no vision — if this matches, the cameras are decoration
- mean-pool instead of ABMIL — whether the attention earned its cost
- vision only, no scalars

## 5. Still outstanding

A clean same-policy test. The 200-episode set trained the head, the 100-episode set chose
its epoch, and the 10k set is a different checkpoint. Fresh 20k rollouts through
`eval_critic.py`, auto-labelled by the sim, are the missing measurement.
