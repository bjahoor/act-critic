# Phase 09 — Critic Head

## 1. Repo layout

`scripts/` held modules and scripts both. Split them.

```
src/
  modeling_act_critic.py
  recording/lerobot_recorder.py
  scripts/
```

Run commands gain `PYTHONPATH=$PWD/src`. Every command in phases 02-08 is updated.

## 2. What the encoder outputs

ResNet18 turns each 200x200 image into a 7x7 grid of 512-dim patches. The encoder's 4 self-attention layers
return the same count.

| Token | Count | |
|---|---|---|
| wrist patches | 49 | 7x7 grid over the image |
| table patches | 49 | |
| `observation.state` | 1 | the 7 arm joints |
| latent `z` | 1 | zeros at inference, so constant across every frame |
| | **100** | 512 each |

This is everything the decoder is given. A missed grasp is the gripper patch and the cube patch being
different patches, so pooling the 98 would destroy the evidence. The head reads all 100; `z` is constant and
learned around.

## 3. Where the head attaches

A parallel branch off the encoder. The decoder is untouched and the trunk is frozen, so the action chunk is
bit-identical with the head attached or removed.

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

Added: the critic head, TCE and ACM. `detach` is the one place a gradient could reach the trunk.

TCE and ACM are arithmetic on the decoder's output, not learned. Only 10 of the 100 planned steps run before
replanning, so consecutive chunks overlap by 90 and TCE reads that overlap. Both enter at the head's output
because two scalars have nowhere to attend.

`failure_score` is squashed to 0-1 inside the head, from a raw value the loss keeps internal. It is not
calibrated and carries no threshold — that is the harness's choice.

## 4. Inside the head

Pool the 100 tokens into one vector, then score it alongside TCE and ACM. Pooling is the whole design; the
scorer is a linear layer either way.

| Pooling | Params | |
|---|---|---|
| mean-pool + MLP | 0.066 M | control |
| **ABMIL gated attention** | **0.033 M** | **picked** |
| 1-query cross-attention | 0.066 M | |
| CLAM | more | ABMIL plus a loss over pseudo-labelled patches |

ABMIL scores each patch and takes a weighted sum. Cross-attention would judge patches against each other,
but the encoder's 4 self-attention layers already did that, so it pays double to redo the frozen trunk's work.
CLAM's extra loss answers attention spreading thin across 10k tiles; there are 100 here. It is the fix if
attention collapses onto one or two patches.

Mean-pool is a control, not a strawman — attentive probing only clearly beats pooling in few-shot settings.
If it ties, attention did not earn its place.

Every option exceeds 200 samples in parameter count. Dropout and early stopping are load-bearing.

[ABMIL](https://proceedings.mlr.press/v80/ilse18a.html) is the standard aggregator in computational
pathology. I found no use of it in robotics.

## 5. Approach

| | |
|---|---|
| Trunk | `bjahoor/act-lift-cube-franka-v2-20k`, frozen |
| Tap | scene tokens, the encoder output |
| Head | ABMIL gated attention pooling, 512 -> 128, linear scorer |
| Extra inputs | TCE and ACM, concatenated at the output |
| Output | `failure_score`, 0-1 |
| Train | all 200 rollouts on 20k |
| Test | the earlier 100 on the same checkpoint, offline |
| Labels | as recorded, episode-wide |
| Imbalance | untouched |

Beat: always-fail, frame index alone, TCE thresholded. Headline metric is earliness.

Failures run to the 500-step giveup and successes finish in ~150, so failures are 69% of frames despite being
42% of episodes. Left alone anyway: 2:1 is mild, thresholding beats reweighting for deep models, and the
threshold sweep is already being run to measure precision and recall. Class weighting is the fix if the head
turns out to answer "failing" everywhere.

Three levels of held-out data. Tuning runs against the earlier 100, seconds per try. The reported number
comes from fresh rollouts with the head in the loop, once, on the frozen design — the sim auto-labels these,
so it is a measurement and not an impression. The 10k set is a third test: a policy the head never saw.

A label derived from `object_pos` would test the labelling rule rather than the head, so `object_pos` and
`termination` are evaluation instruments only.

Open: one frame or a short history, and regularization.
