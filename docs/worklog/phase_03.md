# Phase 03 — LeRobot Install

LeRobot needs to import from the same Python that runs Isaac, with its dependencies kept out of Isaac's. So it goes in
a venv built from Isaac's bundled Python.

## 1. Venv

```bash
~/isaacsim/python.sh -m venv --system-site-packages .venv-lerobot
.venv-lerobot/bin/python -m pip install lerobot==0.4.4
echo .venv-lerobot >> .gitignore
```

`--system-site-packages` is required — Isaac Lab is an editable install, and `.pth` files only work in a real
`site-packages`.

0.4.4 is the newest version on 3.11; 0.5.0 needs 3.12. Rollback is `rm -rf .venv-lerobot`.

## 2. Running

```bash
PYTHONEXE=$PWD/.venv-lerobot/bin/python ~/isaacsim/python.sh scripts/collect_demos.py --enable_cameras --headless
```

`python.sh` puts Isaac's paths ahead of the venv's, so Isaac wins on numpy and torch while lerobot stays importable.

Not `isaaclab.sh` with `VIRTUAL_ENV` — that skips `setup_python_env.sh` and `import isaacsim` fails.

## 3. Test

lerobot imported and a camera rendered in one process, on Isaac's numpy 1.26.0.
