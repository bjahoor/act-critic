# Worklog

| Phase | Title | What it covers |
|---|---|---|
| 00 | [Brainstorm](worklog/phase_00.md) | The idea. One model, one forward pass, an action chunk and a failure score |
| 01 | [Sim and Streaming Setup](worklog/phase_01.md) | Isaac Lab running the lift task, streamed over WebRTC to a headless box |
| 02 | [Record Demos](worklog/phase_02.md) | Cameras added to the scripted expert, and the goal markers turned off |
| 03 | [LeRobot Install](worklog/phase_03.md) | A venv built from Isaac's own Python so both import in one process |
| 04 | [LeRobot Integration](worklog/phase_04.md) | Action space, the recorder, and owning the episode boundary |
| 05 | [Record Demos](worklog/phase_05.md) | 50 successful demos recorded and pushed |
| 06 | [Train ACT](worklog/phase_06.md) | Stock ACT on HF Jobs. Five checkpoints, and two data bugs that cost a run |
| 07 | [Eval Script](worklog/phase_07.md) | Roll out a checkpoint, measure success, save the action chunks |
| 08 | [Failure Data](worklog/phase_08.md) | Rollouts keeping both outcomes — what the head trains on |
| 09 | [Repo Layout](worklog/phase_09.md) | `scripts/` split into modules and things you run |
| 10 | [Critic Head](worklog/phase_10.md) | The design: where it attaches, what it reads, why ABMIL |
| 11 | [The Model File](worklog/phase_11.md) | Building it. The hook, the frozen proof, one bug |
| 12 | [Training Script](worklog/phase_12.md) | Caching the frozen trunk, TCE and ACM, feeding the GPU |
| 13 | [Training Run](worklog/phase_13.md) | The first run, and why its score flatters itself |
| 14 | [Held-Out Measurement](worklog/phase_14.md) | Scoring on episodes never trained on, against a training-free baseline |
| 15 | [Live Score Display](worklog/phase_15.md) | The on-screen readout, streamed |
| 16 | [Live Demo](worklog/phase_16.md) | Running it: three checkpoints, and what the demo does not measure |
