# Phase 10 — The Model File

`src/modeling_act_critic.py`. Design is [phase 09](phase_09.md); this is what it took to build.

## 1. Reaching the encoder output

`ACT.forward` computes the encoder output at
[modeling_act.py:491](../../.venv-lerobot/lib/python3.11/site-packages/lerobot/policies/act/modeling_act.py)
and hands it straight to the decoder. It is never returned.

A forward hook on `model.encoder` takes a copy as it passes, in three lines, rather than reimplementing
`ACT.forward` to return it. LeRobot's file is untouched, so the checkpoint keeps loading and upgrades do not
fork. `vae_encoder` is a separate `ACTEncoder` instance, so hooking `encoder` catches only the right one.

ACT works in `(sequence, batch, dim)`; the head is batch-first. The hook transposes and detaches at the tap.

## 2. Two classes, because they run in different places

`CriticHead` takes tokens as an argument and knows nothing about ACT. Training feeds it cached encoder output,
so the trunk is not merely frozen during training — it is absent.

`ACTWithCritic` subclasses `ACTPolicy` and is the deployed pair. `select_action` and `predict_action_chunk`
are inherited untouched.

## 3. Frozen, and shown to be

`freeze_trunk()` clears `requires_grad` on everything outside `critic.`, and the hook detaches. Measured on
`bjahoor/act-lift-cube-franka-v2-20k`:

| | |
|---|---|
| trainable | 99,332 |
| frozen | 51,601,289 |
| trunk after 20 head steps | bit-identical |
| action chunk after 20 head steps | bit-identical |

The last two are the claim worth making, so they are checked by comparing every trunk tensor before and after
training the head, not asserted.

## 4. Bug

`ACTPolicy.__init__` calls `self.reset()` on its last line. The override clears the history buffer, which the
subclass has not created yet, so construction died before the checkpoint ever loaded. `reset()` guards on
`hasattr`.

## 5. History gate

Four frames at 0, 5, 10 and 15 back is 0.3 s at 50 fps. Until the buffer fills, `critic_score` returns None
rather than padding with copies of the first frame — padding would invent stillness at exactly the point the
approach is being decided.

Cost: no score for the first 0.3 s of an episode. The failure is decided around 1.0-1.8 s, so nothing is lost.

## 6. Attention is returned

`GatedAttentionPool` returns its per-token weights alongside the pooled vector. Drawn over the wrist image
they say what the head learned to watch: the gripper and cube, or a corner of the table. Free, since the
weights are already computed.

`pooling="mean"` swaps in the control, 0.066M against 0.099M, everything else identical.
