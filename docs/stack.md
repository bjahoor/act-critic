# Stack

| Component | Detail |
|---|---|
| Policy | [ACT](https://huggingface.co/docs/lerobot/en/act) via [LeRobot](https://github.com/huggingface/lerobot) 0.4.4, in a venv off Isaac's Python |
| Simulator | Isaac Sim 5.1.0 + Isaac Lab 2.3.2 |
| Collect | `Isaac-Lift-Cube-Franka-IK-Abs-v0` |
| Eval | `Isaac-Lift-Cube-Franka-v0` |
| Python | 3.11.13 |
| numpy / torch | 1.26.0 / 2.13.0+cu130 |
| OS | Ubuntu 24.04 |

**Hardware**

| Where | Role |
|---|---|
| RTX 3060 Ti, 8 GB | Isaac Sim |
| HF Jobs | ACT training |
