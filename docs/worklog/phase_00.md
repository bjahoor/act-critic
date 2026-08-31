# Phase 00 — Brainstorm

ACT with a built-in failure head: one model, one forward pass, an action chunk and a live
`failure_score` together. No code. Written before anything ran, filled in afterwards.

## 1. The idea

Runtime failure detection is usually a second model watching the first — its own encoder, its
own view, inferring from the outside what the policy already knows. Put the detector inside
the policy instead and it reads the policy's own perception directly.

The policy has to fail often enough to be worth detecting, so the plan was an ACT checkpoint
trained deliberately short of convergence.

## 2. What it needs

| | |
|---|---|
| a task that fails | a Franka arm lifting a cube in Isaac Sim, at a mediocre success rate |
| labelled rollouts | both outcomes kept, labelled by the sim itself |
| a frozen trunk | the head must not change the policy — provable, not asserted |
| a signal | the encoder's scene tokens, plus how much the policy's own plan is churning |

## 3. Scope

In: the sim, the demos, ACT training, the head, its training script, the metrics, a live demo.

Out: real hardware, recovery behaviour, anything the score is wired into. The score is a
number the caller reads. Acting on it is somebody else's decision.

## 4. What was left open

Where the head attaches, how it pools 100 tokens into one, how much history it sees, and
whether the whole thing beats a training-free baseline. Phases 10 to 14 answer those.

One weekend, start to finish, in simulation.
