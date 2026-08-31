# act-critic

A weekend proof of concept: ACT with a built-in critic head.

One model, one forward pass, two outputs — the action chunk, and a live score for "am I failing now".

Runtime failure detectors are usually a second model watching the first. This one lives inside it, reading
the policy's own perception rather than guessing at it from the outside.

## Critic head

ACT with a second output path. The head branches off the encoder and scores the same forward pass — no second
model. The trunk is frozen, so the action chunk is bit-identical with the head attached or removed.

[src/modeling_act_critic.py](src/modeling_act_critic.py)

```
  wrist ──┐
          ├──> ResNet18 ──> 98 patches ──┐
  table ──┘     (frozen)                 │
                                         │
  state ──────────────────> 1 token ─────┼──> encoder ──> scene tokens ──┬──> decoder ──> action chunk
                                         │     (frozen)     (100, 512)   │    (frozen)       (100, 9)
  z = 0 ──────────────────> 1 token ─────┘                               │                       │
                                                                      detach                     v
                                                                         │                   TCE, ACM
                                                                         │  ┌───────────────┐    │
                                                                         └─>│  critic head  │<───┘
                                                                            └───────┬───────┘
  TCE  temporal consistency error                                                   │
  ACM  action chunk magnitude                                                       v
                                                                              failure_score
                                                                                0 = fine
                                                                               1 = failing
```

## Inside the critic head

Inside that box. 100 scene tokens and two scalars in, one score out.

Trained by [src/scripts/train_critic.py](src/scripts/train_critic.py)

Scored by [src/scripts/measure_critic.py](src/scripts/measure_critic.py)

```
  ABMIL — how much does each token matter               the head, end to end

    token 1    ──>  0.5%  ┐                             4 frames x 100 tokens (512)
    token 2    ──>  0.3%  │                                          │
       ...                ├──> blend by weight                Linear 512 -> 128
    token 47   ──>   62%  │         │                                │
    token 48   ──>   30%  │         v                             dropout 0.1
    token 100  ──>  0.1%  ┘    pooled (128)                          │
                                                              ABMIL, per frame
    47 is the gripper, 48 the cube.                                  │
    An average gives all 100 an equal 1%.                 4 x 128 concatenated
                                                                     │
    scoring one token:                                    + TCE + ACM  = 514
        ──> tanh(V·) ─┐                                              │
                      ⊙ ──> one number                        Linear 514 -> 1
        ──> sigm(U·) ─┘  then softmax over 100                       │
            the gate                                               sigmoid
                                                                     │
                                                               failure_score
```

ABMIL — attention-based multiple instance learning — scores every token, softmaxes the scores into weights
summing to 1, and blends the tokens by them.
Mean pooling is the same operation with every weight fixed at 1/100.

## Worklog

| | | |
|---|---|---|
| 00 | [Brainstorm](docs/worklog/phase_00.md) | The idea. One model, one forward pass, an action chunk and a failure score |
| 01 | [Sim and Streaming Setup](docs/worklog/phase_01.md) | Isaac Lab running the lift task, streamed over WebRTC to a headless box |
| 02 | [Record Demos](docs/worklog/phase_02.md) | Cameras added to the scripted expert, and the goal markers turned off |
| 03 | [LeRobot Install](docs/worklog/phase_03.md) | A venv built from Isaac's own Python so both import in one process |
| 04 | [LeRobot Integration](docs/worklog/phase_04.md) | Action space, the recorder, and owning the episode boundary |
| 05 | [Record Demos](docs/worklog/phase_05.md) | 50 successful demos recorded and pushed |
| 06 | [Train ACT](docs/worklog/phase_06.md) | Stock ACT on HF Jobs. Five checkpoints, and two data bugs that cost a run |
| 07 | [Eval Script](docs/worklog/phase_07.md) | Roll out a checkpoint, measure success, save the action chunks |
| 08 | [Failure Data](docs/worklog/phase_08.md) | Rollouts keeping both outcomes — what the head trains on |
| 09 | [Repo Layout](docs/worklog/phase_09.md) | `scripts/` split into modules and things you run |
| 10 | [Critic Head](docs/worklog/phase_10.md) | The design: where it attaches, what it reads, why ABMIL |
| 11 | [The Model File](docs/worklog/phase_11.md) | Building it. The hook, the frozen proof, one bug |
| 12 | [Training Script](docs/worklog/phase_12.md) | Caching the frozen trunk, TCE and ACM, feeding the GPU |
| 13 | [Training Run](docs/worklog/phase_13.md) | 0.99 episode AP, and why that number is optimistic |
| 14 | [Held-Out Measurement](docs/worklog/phase_14.md) | The honest numbers: 94% caught, 4% false alarms, first alarm at 1.86 s |
| 15 | [Live Score Display](docs/worklog/phase_15.md) | The on-screen readout, streamed |
| 16 | [Live Demo](docs/worklog/phase_16.md) | Running it: three checkpoints, and what the demo does not measure |

See [docs/stack.md](docs/stack.md) for versions and hardware.
