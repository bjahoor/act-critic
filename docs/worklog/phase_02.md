# Phase 02 — Record Demos

Modify `src/scripts/collect_demos.py` to record 50 successful episodes with randomized cube positions.

## 1. Copy the state machine into the repo

Isaac Lab stays read-only, so the scripted expert is copied out to be modified here.

```bash
cp ~/IsaacLab/scripts/environments/state_machine/lift_cube_sm.py src/scripts/collect_demos.py
```

BSD-3-Clause header kept. Records nothing yet.

## 2. Verify the copy still runs

```bash
LIVESTREAM=1 PUBLIC_IP=<server-ip> PYTHONPATH=$PWD/src ~/IsaacLab/isaaclab.sh -p \
  src/scripts/collect_demos.py --num_envs 8
```

Same behaviour as the original before any changes are made.

## 3. Add cameras

The lift environment is state-only. ACT needs pixels.

`wrist_cam` and `table_cam` are lifted from Isaac Lab's Franka visuomotor config
(`manager_based/manipulation/stack/config/franka/stack_ik_rel_visuomotor_env_cfg.py`), with `distance_to_image_plane`
dropped from `data_types` — ACT uses RGB.

Isaac Lab is read-only, so they are set on the config object in `collect_demos.py` between `parse_env_cfg()` and
`gym.make()`.

[Camera sensor](https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/camera.html)

## 4. Changes to `collect_demos.py`

- `AppLauncher(args_cli)` replaces `AppLauncher(headless=args_cli.headless)`. The original only forwards `headless`, so
  `--enable_cameras` was parsed and discarded.
- Imports added: `mdp`, `sim_utils`, `ObsTerm`, `SceneEntityCfg`, `CameraCfg`.
- `wrist_cam` on `panda_hand` and `table_cam` overhead, both 200x200 RGB.
- Both registered as observations via `mdp.image`, plus `concatenate_terms = False` — images can't be flattened into the
  state vector.

The state machine itself is untouched.

## 5. Test

```bash
LIVESTREAM=1 PUBLIC_IP=<server-ip> PYTHONPATH=$PWD/src ~/IsaacLab/isaaclab.sh -p \
  src/scripts/collect_demos.py --num_envs 4 --enable_cameras
```

Camera prims are visible in the streamed viewport. A temporary print of the observation shapes gave
`(2, 200, 200, 3) uint8` for both cameras on a 2-env run, confirming the images reach the observation manager and not
just the scene. Removed afterwards.

## 6. Pull the cameras back

Both views were too tight. Offsets widened, rotations unchanged.

| Camera | Before | After |
|---|---|---|
| `wrist_cam` | `(0.13, 0.0, -0.15)` | `(0.13, 0.0, -0.3)` |
| `table_cam` | `(1.0, 0.0, 0.4)` | `(1.4, 0.0, 0.7)` |

`clipping_range` far plane stays at 2.0 m, still well beyond both.

## 7. Turn off the debug markers

The goal-pose command draws an axis triad in the scene (`debug_vis=True` in `CommandsCfg`). It is a rendered prim, so it
would appear in the recorded camera images.

```python
env_cfg.commands.object_pose.debug_vis = False
```

Set in `src/scripts/collect_demos.py`, not in Isaac Lab. The end-effector frame transformer already has `debug_vis=False`.

## 8. Test

```bash
LIVESTREAM=1 PUBLIC_IP=<server-ip> PYTHONPATH=$PWD/src ~/IsaacLab/isaaclab.sh -p \
  src/scripts/collect_demos.py --num_envs 4 --enable_cameras
```

Wider framing on both cameras, no axis triads in the scene.
