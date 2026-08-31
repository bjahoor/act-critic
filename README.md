# act-critic

Over one weekend I built a proof of concept: a modified ACT architecture with a built-in critic head. Start
to finish, trained and tested in simulation.

The task is a Franka Panda arm lifting a cube to a fixed point in Isaac Sim. The policy learns it from
scripted demonstrations, and lands at a success rate that is deliberately mediocre — it has to fail often
enough to give the critic head something to detect.

Runtime failure detectors are usually a second model watching the first. This one lives inside the policy,
reading its own perception rather than inferring from the outside.

---

## Runtime Demo

The live demo, streamed out of Isaac Sim. Recorded runs are in [docs/videos/](docs/videos/).

![The gripper closed just above the cube; the failure score reads 0.80](docs/images/live_demo.png)

The gripper closed just above the cube. The score climbs once the grasp misses.

---

## Runtime Loop

What happens on every step at runtime.

```
                      2 cameras 200x200 RGB  +  7 arm joints
        ┌─────────────────────────────────────────────────┐
        │                                                 v
  ┌─────┴───────────┐                       ┌─────────────────────────────────┐
  │    Isaac Sim    │                       │        ACT + critic head        │
  │  lift the cube  │                       │       ACT weights frozen        │
  │ 50 steps / sec  │                       │      one pass, two outputs      │
  └─────┬───────────┘                       └───────┬───────────────────┬─────┘
        │                                           │                   │
        │     9 joint targets                       │                   │
        └───────────────────────────────────────────┘                   │
          100-step chunk, first 10 executed                             v
                                                                  failure_score
                                                                   0.00 - 1.00
                                                                        │
                                                                        v
                                                                 ┌─────────────┐
                                                                 │ score panel │
                                                                 └──────┬──────┘
                                                                        │
   viewport + panel ──> WebRTC ──> remote viewer <──────────────────────┘
```

---

## Critic Head

My modified ACT architecture. The critic head I added branches off the transformer encoder's output.

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

---

## Inside the Critic Head

Inside that box. 100 scene tokens and two scalars in, one score out.

Trained by [src/scripts/train_critic.py](src/scripts/train_critic.py)

Scored by [src/scripts/measure_critic.py](src/scripts/measure_critic.py)

```
                          ┌─ critic head ───────────────────────────┐
  4 frames x 100 tokens   │                                         │
  (512) ─────────────────>┼- - - - ->  Linear 512 -> 128            │
                          │                    │                    │
                          │               dropout 0.1               │
                          │                    │                    │
                          │            ABMIL, per frame             │
                          │                    │                    │
                          │          4 x 128 concatenated           │
                          │                    │                    │
  TCE + ACM ─────────────>┼- - - - ->  + TCE + ACM = 514            │
                          │                    │                    │
                          │             Linear 514 -> 1             │
                          │                    │                    │
                          │                 sigmoid                 │
                          └────────────────────┬────────────────────┘
                                               v
                                         failure_score
```

The tokens go through the whole stack; TCE and ACM skip it and join just before the score.

```
  ABMIL — how much does each token matter

    token 1    ──>  0.5%   ┐
    token 2    ──>  0.3%   │
       ...                 ├──> blend by weight ──> pooled (128)
    token 47   ──>   62%   │
    token 48   ──>   30%   │
    token 100  ──>  0.1%   ┘

    47 is the gripper, 48 the cube.  An average gives all 100 an equal 1%.

    scoring one token:
        ──> tanh(V·) ─┐
                      ⊙ ──> one number, then softmax over 100
        ──> sigm(U·) ─┘
            the gate
```

ABMIL — attention-based multiple instance learning — scores every token, softmaxes the scores into weights
summing to 1, and blends the tokens by them.
Mean pooling is the same operation with every weight fixed at 1/100.

---

The full build, phase by phase: [docs/worklog.md](docs/worklog.md).

See [docs/stack.md](docs/stack.md) for versions and hardware.
