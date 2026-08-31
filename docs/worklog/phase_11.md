# Phase 11 — The Model File

`src/modeling_act_critic.py`. Design is [phase 10](phase_10.md); this is what it took to build.

## 1. Reaching the encoder output

The encoder output is handed straight to the decoder and never returned. A forward hook on `model.encoder`
takes a copy as it passes, so LeRobot's file stays untouched and upgrades do not fork. `vae_encoder` is a
separate instance, so the hook catches only the right one.

## 2. Two classes

`CriticHead` takes tokens as an argument and knows nothing about ACT — training feeds it cached output, so
the trunk is absent rather than merely frozen. `ACTWithCritic` is the deployed pair.

## 3. The head

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

The score comes from two branches — `tanh(V)` for what is in the token, `sigmoid(U)` as a gate — multiplied
together. The gate is what makes it *gated*: `tanh` alone is near-linear around zero and cannot suppress a
dimension outright.

Frames are pooled separately and concatenated, so the ordering survives.

## 4. Frozen, and shown to be

| | |
|---|---|
| trainable | 99,332 |
| frozen | 51,601,289 |
| trunk after 20 head steps | bit-identical |
| action chunk after 20 head steps | bit-identical |

The last two are the claim worth making, so they are checked tensor by tensor rather than asserted.

## 5. Bug

`ACTPolicy.__init__` calls `reset()` on its last line, before the subclass has built its history buffer.
Construction died before the checkpoint ever loaded. `reset()` guards on `hasattr`.

## 6. Two choices in the code

Until the history fills, `critic_score` returns None rather than padding with copies of the first frame —
padding would invent stillness exactly where the approach is being decided. Costs the first 0.3 s of an
episode; the failure is decided around 1.0-1.8 s.

`GatedAttentionPool` returns its per-token weights. Drawn over the wrist image they say whether the head
learned to watch the gripper or a corner of the table. Free, since they are already computed.
