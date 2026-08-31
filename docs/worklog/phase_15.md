# Phase 15 — Live Score Display

`src/scripts/eval_critic.py`. The demo: ACT running with the head attached, failure score on screen as it happens.

Copy of [phase 07](phase_07.md)'s eval loop. Same env, same task. Recording is dropped — `eval_policy.py` owns that.

```bash
LIVESTREAM=1 PUBLIC_IP=<server-ip> PYTHONEXE=$PWD/.venv-lerobot/bin/python ~/isaacsim/python.sh \
  src/scripts/eval_critic.py --model 20k --num_envs 4 --enable_cameras
```

## 1. The panel

A Kit window: one bar per env, plus a rolling chart of the last 200 steps.

| | |
|---|---|
| `omni.ui` | Isaac Sim's own toolkit, already installed |
| `isaaclab.ui.widgets.LiveLinePlot` | Isaac Lab's rolling multi-series plot |

Both ship with the install. Nothing added.

It reaches a remote viewer because the livestream carries the whole application window, and Isaac Lab
un-hides the UI specifically when livestreaming. Isaac Lab builds its own window this way, so the pattern
already renders under our launch.

Built only when `sim.has_gui()`, so a plain headless run is unaffected.

`LiveLinePlot` must be constructed inside the parented frame. Its `__init__` reads attributes assigned after
it, and works at all only because `ui.Frame(build_fn=...)` builds lazily.

## 2. Not in the scene

A bar drawn above the robot would be simpler and is the obvious idea. It lands in `wrist_cam` and
`table_cam`, which are the policy's observations, so the readout would change the behaviour it is reporting
on. Same reason the goal marker is off in [phase 07](phase_07.md).

`debug_draw` was the other candidate. It has no text at all — points and lines only.

## 3. TCE and ACM

The head takes two scalars beside the tokens, and they must arrive on the scale training used or the head
reads a number it has never seen, silently.

Both are recomputed here frame by frame, mirroring `train_critic.compute_tce_acm`. The scaler is loaded from
`critic.pt`, where training saved it beside the weights. The checkpoint's own action mean and std normalize
the chunk first.

Without a trained head the script still runs — random weights, noise on the bars, enough to check the
display.

## 4. Known cost

The trunk runs twice per env per step. TCE and ACM come from this frame's chunk, so the chunk is needed
before the head can be called, and `critic_score` predicts it again internally.

Correct, not free. The fix is to move the calculation inside the head, which already keeps a frame history.
