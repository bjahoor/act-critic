# Phase 12 — Training Script

`src/scripts/train_critic.py`. Cache the frozen trunk's output once, then train the head on it.

## 1. Why a cache

The trunk never changes, so its output for a frame is fixed. Decoding 200 videos every epoch would dominate
the run; decoding once and writing the encoder output does not. It also makes the stop-gradient claim
structural — ACT is not loaded during head training at all.

One file per episode, `float16`, ~30 MB per 500-step episode. Both caches together are ~10 GB, keyed by
dataset and checkpoint so the transfer set cannot silently reuse the training set's.

## 2. TCE and ACM

Computed from the saved chunks, normalized with the checkpoint's own action statistics. Raw, the arm's
radians dwarf the fingers' metres. TCE compares a chunk against the one 10 steps back, over the 90 they
overlap — adjacent chunks differ mostly by noise, and 10 is the real replan interval.

Verified: a policy repeating one plan gives TCE exactly 0, one replanning with unit noise gives 2.

## 3. Bug found by review — the deployed head saw 2 frames, not 4

`critic_score` read whatever the encoder hook last caught, but `select_action` only runs the encoder every
`n_action_steps`, which is 10. So the four history frames were duplicates of two, 0.2 s apart, while training
used four distinct frames from `predict_action_chunk` on every frame.

Training and deployment disagreeing on the input, silently, and it would have shown up only as the head
underperforming in the live eval. `critic_score` now runs the trunk itself and returns the chunk, so the
caller takes TCE and ACM from it rather than paying for a second pass.

## 4. Metric

Frame AP is nearly saturated by episode length: failures run to the 500-step giveup, successes stop at ~150.
Measured on the tune cache, frame index alone scores **0.913**. Early stopping watches episode AP instead —
the mean score over an episode against its label — where there are 100 samples, not 29,221.

Baselines printed at startup, so a result is never read without them:

| | |
|---|---|
| always-fail, episode AP | 0.420 |
| always-fail, frame AP | 0.696 |
| frame index, frame AP | 0.913 |
| TCE alone, episode AP | 0.829 |

TCE alone is the bar.

## 5. Feeding the GPU

The head is 0.1M parameters and runs at 25,000 samples/s in 1.1 GB of the 3060 Ti's 8. The data path, not the
GPU, is the constraint: each sample is four frames of 100x512.

| | before | after |
|---|---|---|
| throughput | 2,155 /s | 4,040 /s |
| across PCIe | 819 KB, float32 | 410 KB, float16 |
| anonymous RAM, one epoch | +9.6 GB | +0.89 GB |

`.npz` was the cause of the memory: `mmap_mode` is silently ignored on it, so every worker built its own
uncompressed copy. Tokens are a bare `.npy` now, memory-mapped and shared. The cast to float32 moved into the
head, after the transfer.

The GPU is still idle most of the time. It does not matter — an epoch is ~7 s and the whole run is minutes.

## 6. Unverified

The cache is built from AV1-encoded video; the robot sees raw sim RGB. Frozen ResNet features are not
invariant to that, and the source frames were deleted after encoding, so it could not be measured. Keeping a
few hundred raw frames on the next recording would settle it.
