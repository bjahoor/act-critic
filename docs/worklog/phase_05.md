# Phase 05 — Record Demos

## 1. Record

```bash
LIVESTREAM=1 PUBLIC_IP=<server-ip> PYTHONEXE=$PWD/.venv-lerobot/bin/python ~/isaacsim/python.sh \
  scripts/collect_demos.py --num_envs 10 --enable_cameras --record --num_demos 100 --overwrite
```

100 episodes, 14693 frames, 140-159 steps each. No actions outside joint limits. 10 envs peaked at 4.9 GB of 8; wall
clock is dominated by inline AV1 encoding, not the sim.

## 2. Push to hub

`scripts/push_dataset.py` refuses to run on:

- a `--root` that is not a dataset
- a truncated or unfinalized dataset
- an existing repo, without `--force`

Uploads the whole directory, so keep stray files out.

```bash
.venv-lerobot/bin/python scripts/push_dataset.py --repo_id bjahoor/lift-cube-franka
```

[LeRobot](https://huggingface.co/docs/lerobot/en/index)
