# Phase 01 — Sim and Streaming Setup

## 1. Symlink Isaac Lab into the repo

Isaac Sim and Isaac Lab stay where they were installed, read-only. The repo reaches Isaac Lab through a gitignored symlink.

```bash
ln -s ~/IsaacLab IsaacLab
echo IsaacLab >> .gitignore
```

## 2. Run the sim, streamed over WebRTC

Headless machine over SSH, so the viewport is streamed.

```bash
cd IsaacLab
LIVESTREAM=1 PUBLIC_IP=<server-ip> ./isaaclab.sh -p scripts/environments/state_machine/lift_cube_sm.py --num_envs 8
```

Isaac Sim WebRTC Streaming Client connects to the same IP. 49100/TCP signaling, 47998/UDP media.

[Livestream clients](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/manual_livestream_clients.html) · [client download](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html) · [AppLauncher](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.app.html)

## 3. What the script does

`IsaacLab/scripts/environments/state_machine/lift_cube_sm.py`

A hardcoded state machine, not a learned policy. Five states, looped:

1. `REST` — hold still
2. `APPROACH_ABOVE_OBJECT` — move 10 cm above the cube
3. `APPROACH_OBJECT` — descend onto it
4. `GRASP_OBJECT` — close the gripper
5. `LIFT_OBJECT` — carry it to the commanded pose

Each transition waits for the end-effector to come within 1 cm of its target, then holds for a fixed dwell time. Runs as a warp kernel, so all envs step in parallel.

It reads the cube pose directly from the simulator rather than from cameras, and it records nothing. The environment randomizes the cube position on every reset, so demos vary without any change.
