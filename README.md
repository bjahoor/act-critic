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

See [docs/stack.md](docs/stack.md) for versions and hardware.
