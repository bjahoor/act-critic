# act-critic

ACT with a built-in failure head — one model, one forward pass, outputs both an action chunk and a live "am I failing now" score.

Most runtime failure detectors run as a second model alongside the policy. This one lives inside it.

I wanted to build it over a weekend.

## Stack

| | |
|---|---|
| Policy | ACT via [LeRobot](https://github.com/huggingface/lerobot) 0.4.4 |
| Simulator | Isaac Sim 5.1.0 + Isaac Lab 2.3.2 |
| Task | `Isaac-Lift-Cube-Franka-IK-Rel-v0` |
| Python | 3.12.3 |
| OS | Ubuntu 24.04 |

**Hardware**

| | |
|---|---|
| RTX 3060 Ti, 8 GB | Isaac Sim |
| RTX 3060, 12 GB | ACT training, inference |
