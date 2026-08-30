# Phase 04 — LeRobot Integration

## 1. Action space

Absolute joint positions, ACT's default.

| Field | Source | Shape |
|---|---|---|
| `observation.state` | `robot.data.joint_pos` | 9 |
| `action` | `robot.data.joint_pos_target` | 9 |

State captured before the step, target after.

[Action representations](https://huggingface.co/docs/lerobot/en/action_representations)

Two tasks, because no scripted expert exists for the joint-command one.

```
 COLLECT ── Isaac-Lift-Cube-Franka-IK-Abs-v0

   ┌───────────────┐  gripper pose   ┌────┐  joint targets   ┌────────┐
   │ state machine │ ──────────────> │ IK │ ───────┬───────> │ motors │
   └───────────────┘                 └────┘        │         └────────┘
                                                   v
                                              recorded as `action`

 EVAL ───── Isaac-Lift-Cube-Franka-v0

   ┌─────┐          joint targets                            ┌────────┐
   │ ACT │ ────────────────────────────────────────────────> │ motors │
   └─────┘                                                   └────────┘
```

Eval must override the robot to `FRANKA_PANDA_HIGH_PD_CFG`, which `IK-Abs` sets and it does not.

## 2. Recorder

`scripts/lerobot_recorder.py`. No Isaac imports, runs from the venv in [phase 03](phase_03.md).

Buffers frames per env, writes only successes. Two 200x200 videos, a 9-vector state and action, 50 fps.

## 3. Own the episode boundary

Isaac resets inside `env.step()` before returning, so a success check afterwards sees a teleported cube. Both
terminations are taken over instead.

```python
success_term = DoneTerm(func=lift_mdp.object_reached_goal, params={"threshold": 0.1})
dropped_term = env_cfg.terminations.object_dropping
env_cfg.terminations.time_out = None
env_cfg.terminations.object_dropping = None
```

0.1 is generous on purpose — the failures worth detecting are dropped or missed grasps, not a few cm short.

## 4. Manual reset

Episodes end on a 0.5 s hold at the goal, a dropped cube, or a 500-step giveup. Only that env resets.

500 steps is 10 s. The expert finishes in ~150; the headroom is for a slower ACT policy.

## 5. Record

- First step after a reset is skipped, it mixes two episodes.
- Actions clamped to `joint_pos_limits`; the IK can ask for angles the joints do not have.
- Startup assert on `joint_names` order.
- `recorder.close()` after the loop, `atexit` as backstop. Without `finalize()` the dataset cannot be loaded.
