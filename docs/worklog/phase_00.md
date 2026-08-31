# Phase 00 — Brainstorm

No code. Design only.

## 1. The idea

ACT with a built-in critic head: one model, one forward pass, an action chunk and a live
`failure_score` together.

Runtime failure detection is usually a second model watching the first. Inside the policy, it
reads the policy's own perception instead.

## 2. Scope

A Franka arm lifting a cube in Isaac Sim, at a success rate low enough to give the head
something to detect. The score is a number the caller reads — acting on it is out of scope.
