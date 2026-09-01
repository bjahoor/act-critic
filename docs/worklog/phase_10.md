# Phase 10 — Critic Head

## 1. What the encoder outputs

ResNet18 turns each 200x200 image into a 7x7 grid of 512-dim patches. The encoder's 4 self-attention layers
return the same count.

| Token | Count | |
|---|---|---|
| wrist patches | 49 | 7x7 grid over the image |
| table patches | 49 | |
| `observation.state` | 1 | the 7 arm joints |
| latent `z` | 1 | zeros going in; the encoder mixes the other tokens into it |
| | **100** | 512 each |

This is everything the decoder is given. A missed grasp is the gripper patch and the cube patch being
different patches, so pooling the 98 would destroy the evidence. The head reads all 100.

## 2. Where the head attaches

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

## 3. Inside the head

Pool the 100 tokens into one vector, then score it alongside TCE and ACM. Pooling is the whole design; the
scorer is a linear layer either way.

| Pooling | Params | |
|---|---|---|
| mean-pool + MLP | 0.066 M | control |
| **ABMIL gated attention** | **0.033 M** | **picked** |
| 1-query cross-attention | 0.066 M | |
| CLAM | more | ABMIL plus a loss over pseudo-labelled tokens |

ABMIL — attention-based multiple instance learning — scores each token and takes a weighted sum.
Cross-attention would judge tokens against each other, but the encoder's 4 self-attention layers already did
that, so it pays double to redo the frozen trunk's work.

CLAM's extra loss answers attention spreading thin across 10k tiles; there are 100 tokens here. It is the fix
if attention collapses onto one or two tokens.

Mean-pool is a control, not a strawman — attentive probing only clearly beats pooling in few-shot settings.
If it ties, attention did not earn its place.

Every option exceeds 200 samples in parameter count. Dropout and early stopping are load-bearing.

[ABMIL](https://proceedings.mlr.press/v80/ilse18a.html) is the standard aggregator in computational
pathology. I found no use of it in robotics.

## 4. Approach

| | |
|---|---|
| Trunk | `bjahoor/act-lift-cube-franka-v2-20k`, frozen |
| Tap | scene tokens, the encoder output |
| Head | ABMIL gated attention pooling, 512 -> 128, linear scorer |
| Extra inputs | TCE and ACM, concatenated at the output |
| Output | `failure_score`, 0-1 |
| History | four frames: now, 0.1 s, 0.2 s, 0.3 s back |
| Labels | as recorded, episode-wide |
| Imbalance | untouched |
| Regularization | dropout 0.1, early stopping on held-out average precision |

Dropout 0.1 is ACT's own value rather than one picked here. Early stopping watches average precision because
accuracy is gameable at 24/76. Four frames rather than one so a descent toward the cube is distinguishable
from one past it.

## 5. Data

| | | |
|---|---|---|
| Train | `bjahoor/lift-cube-rollouts-20k-200` | 98 success / 102 failure |
| Tune against | `bjahoor/lift-cube-rollouts-20k` | the earlier 100, seconds per try |
| Transfer test | `bjahoor/lift-cube-rollouts-10k` | a checkpoint the head never saw |

The reported number comes from fresh rollouts with the head in the loop, once, on the frozen design. The sim
auto-labels those, so it is a measurement and not an impression.

Every failure in all three sets is a timeout; there are no drops, because the drop check only fires when the
cube leaves the table and this policy fails by missing the grasp. The head is trained and evaluated on
exactly one failure mode. Nothing here supports a claim of generality.

## 6. Imbalance, left alone

Failures run to the 500-step giveup and successes finish in ~150, so failures are 76% of frames despite being
51% of episodes. 3:1 is mild, thresholding beats reweighting for deep models, and the threshold sweep is
already being run to measure precision and recall. Class weighting is the fix if the head answers "failing"
everywhere.

## 7. What it could cheat on

The cube position is randomized per episode and plainly visible, so the head could learn "cube at this spot
failed" and never look at the gripper. The signature is a gap between training and held-out scores.

A label derived from `object_pos` would test the labelling rule rather than the head, so `object_pos` and
`termination` are evaluation instruments only.

Beat: always-fail, frame index alone, TCE thresholded. Headline metric is earliness — precision and recall
alone look fine for a head that fires only once the episode is visibly lost.
