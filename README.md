# act-critic

A weekend proof of concept: ACT with a built-in critic head.

One model, one forward pass, two outputs — the action chunk, and a live score for "am I failing now".

Runtime failure detectors are usually a second model watching the first. This one lives inside it, reading
the policy's own perception rather than guessing at it from the outside.

## Critic head

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

See [docs/stack.md](docs/stack.md) for versions and hardware.
