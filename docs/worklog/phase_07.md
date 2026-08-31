# Phase 07 — Eval Script

Measure a trained checkpoint's success rate, and write the labelled rollouts the failure head trains on.

```bash
cp src/scripts/collect_demos.py src/scripts/eval_policy.py
```

Everything below is a change to that copy, except step 1.

## 1. Recorder

`lerobot_recorder.py` gains `keep_failures`, default off. On, it declares a `failure` field and writes every episode.
Off, `collect_demos.py` is unchanged.

1 is failure, 0 is success — the head predicts failure, so the label matches its output.

Rollout mode also stores the cube position per frame and why the episode ended. Without them the label is
episode-wide, and when a failure began cannot be recovered without re-running.

State and action dims are separate now — 7 arm joints in, 9 joint targets out.

## 2. Task

`Isaac-Lift-Cube-Franka-v0`, the joint-command task. ACT emits the 9-vector it was trained on, the env takes it
directly.

```python
env_cfg.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
env_cfg.actions.arm_action = mdp.JointPositionActionCfg(
    asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=False
)
env_cfg.actions.gripper_action = None
```

One term over all nine joints, so the order matches `joint_pos_target`. The stock task splits arm and a binary
gripper, which does not.

`HIGH_PD` because `IK-Abs` sets it and this task does not.

The goal is the fixed point from [phase 04](phase_04.md), and resampling is disabled — the default 5 s would fall
inside an episode here.

The demos only ever held the fingers fully open or fully closed, so the policy regresses the midpoint when unsure.
Predictions below 0.03 are snapped closed, the rest open.

## 3. Policy

The state machine and its warp kernel are deleted. `--model` picks a checkpoint.

| Flag | Repo |
|---|---|
| `10k` .. `40k` | `bjahoor/act-lift-cube-franka-v2-10k` .. `-40k` |
| `50k` | `bjahoor/act-lift-cube-franka-v2` |

One policy instance per env — ACT's action queue is shared across the batch with no per-env reset. Identical weights,
~200 MB each.

## 4. Chunks

`select_action` only predicts every `n_action_steps`. TCE needs one per step, so `predict_action_chunk` is called
alongside it, one chunk per recorded frame, saved as one `.npy` per episode. Written beside the dataset, not inside
it — `push_dataset.py` uploads the whole directory.

## 5. Run

Success rate only.

```bash
LIVESTREAM=1 PUBLIC_IP=<server-ip> PYTHONEXE=$PWD/.venv-lerobot/bin/python PYTHONPATH=$PWD/src ~/isaacsim/python.sh \
  src/scripts/eval_policy.py --model 50k --num_envs 4 --num_rollouts 50 --enable_cameras
```

Rollouts for the head.

```bash
LIVESTREAM=1 PUBLIC_IP=<server-ip> PYTHONEXE=$PWD/.venv-lerobot/bin/python PYTHONPATH=$PWD/src ~/isaacsim/python.sh \
  src/scripts/eval_policy.py --model 30k --num_envs 4 --num_rollouts 300 --enable_cameras \
  --record --dataset_root datasets/rollouts --overwrite
```
