# Idea

Origin, alternatives, outcome, next.

## 1. The problem

Physical AI gets deployed on real robots. There will always be a better model next month, with a better
success rate, and it will never be 100%. Something always fails. The gap does not close by waiting for the
next checkpoint.

So the question is how to close it around the model instead of inside it. Two sub-problems:

| sub-problem | what it means |
|---|---|
| **Close the gap** | when a rollout is going wrong, attempt to recover it — steer, retry, replan |
| **Hand it over** | when it cannot be recovered, pass control to a human, teleoperated |

Both need the same thing first: the robot has to know it is failing, while it is still failing. Nothing here
is specific to any one policy yet — this is the problem for physical AI in general.

## 2. Detect failure, then fork

Nothing can be done about a failure that has not been noticed. Detection is not one of the options — it is
the thing both options are built on.

```
  rollout running
        │
        v
  recognize we are failing        <- everything below depends on this
        │
        v
  can this be recovered?
        │
        ├── yes ──> close the gap ──> steer / retry / replan
        │                 │
        │              failed
        │                 │
        └── no ───────────┴──> hand over to teleop
```

So detection is the piece to build first, and the piece the other two are useless without.

## 3. How to recognize failure

| heuristic | learned |
|---|---|
| threshold something computable — plan disagreement, correction size, elapsed steps | train on rollouts that ended well and rollouts that did not — a head on the policy, or a VLM from scratch |

Heuristics only read the plan and the clock. Most failures are visible in the scene, which they never see.
A rule also only catches the failure it was written for, where learning generalizes to modes nobody listed.

The heuristic stays as the baseline anything learned has to beat.

## 4. What already exists

| approach | kind | what it does |
|---|---|---|
| [Sentinel](https://arxiv.org/abs/2410.04640) | both | temporal action consistency, plus a VLM asking whether the task is progressing |
| [SAFE](https://arxiv.org/abs/2506.09937) | learned | a scalar read off a VLA's internal features, trained on successes and failures |
| a VLM watching the camera | learned | prompt it every N frames. No training, but slow, and it sees only what a bystander sees |
| plan disagreement | heuristic | threshold how much consecutive plans differ. Free |

The learned ones all read something internal to the policy. That is the pattern worth copying — the policy
already computed the perception, and reading it costs nothing extra.

## 5. Choosing the policy

| policy | what it is |
|---|---|
| **VLA** | what SAFE reads. Generalist, many tasks, language-conditioned |
| **ACT** | one task, small, and it emits a plan — a hundred steps, replanned constantly |

A VLA is the better subject and the wrong one for a weekend. Fine-tuning one, then rolling out enough
failures to train a detector, is a compute and time budget this did not have.

ACT instead. It fits one GPU, and the chunk gives the detector something a single-action policy cannot: a
plan to inspect before it runs, and a previous plan to compare it against.

## 6. Two choices

**Learned, not heuristic.** A rule only catches the failure it was written for, and it only reads the plan
and the clock. Learning generalizes, and it can read the scene.

**Inside the model, not beside it.** Every detector above is a second model watching the first. This one is a
head on the policy, reading the perception the policy already computed. One forward pass, two outputs — so
it is nearly free, it sees the policy's own features rather than a bystander's view, and it answers at
control rate.

## 7. What to read, what to emit

ACT locked in, learning locked in. What is actually available to tap:

| tap | what it is |
|---|---|
| the camera feeds | raw pixels, before the backbone touches them |
| `z` | the latent. Zeros at inference, but the encoder mixes the scene into it |
| the current state | the arm's joint positions |
| the scene tokens | the encoder's output — the perception, after self-attention |
| the action chunk | the decoder's output. The plan, a hundred steps of it |

The chunk is worth more than it looks. It is replanned every ten steps, so consecutive plans overlap by
ninety and can be compared: how much they disagree, and how large the corrections are. Two numbers, free,
no training.

The output is a scalar from 0 to 1. Higher means more likely to be failing right now. What to do with that
number is another layer's problem.

## 8. Where the head goes

| option | what it is | verdict |
|---|---|---|
| MLP on pooled features | blend the 100 tokens into one vector, then a small feed-forward net on it | rejected — pooling averages away local evidence |
| extra decoder query | add a 101st query to ACT's own decoder and read its answer | cheap, but borrows the action path's attention |
| dedicated cross-attention block | one new attention layer, its own weights, querying the scene tokens | picked in the plan — own attention, can look where the action path does not |
| separate decoder | a second decoder stack beside ACT's | rejected — most parameters, most risk, least gain |
| any of the above, unfrozen | let the critic's loss reach the trunk so its features shape for failure too | rejected — it degrades the policy, and the critic loss would tune 51.6 M parameters on 200 rollouts and just memorize them |

## 9. Why the cross-attention block lost

The plan picked it. The build did not, for two reasons.

**Cost.** Cross-attention needs three projection matrices. Whether you tap the encoder's output or the raw
patches, that layer is there — no way around it, and nothing extra bought for it.

**The question it can ask.** The query never changes, so cross-attention scores linearly. Gated attention
scores using two curves.

Failure isn't "more of something is worse." It's a specific situation — close to the cube but not closing.
A line can only say more or less. A curve can say *this exact condition*. For this reason, we selected gated
attention.

## 10. ABMIL

The pooling method is [ABMIL](https://arxiv.org/abs/1802.04712) — attention-based multiple instance
learning. We chose its gated version, where a `sigmoid` branch multiplies the `tanh` branch. That multiply is
what produces the nonlinear scoring of section 9.

It also pools, which is the other thing the head needs. The weights normalize to sum to 1 and blend the
hundred tokens into one vector, ready for a linear layer to turn into the scalar. Better than mean or max
pooling because the weighting is learned.

## 11. What it buys, what it costs

| pro | why |
|---|---|
| the policy is untouched | frozen trunk, so the chunk is bit-identical with the head attached or removed |
| nearly free | one forward pass, two outputs, at control rate |
| it sees what the policy sees | the policy's own perception, not a bystander's view |
| cheap to train | only the head learns |

| con | why |
|---|---|
| the features are not the head's | the trunk was trained to predict actions, not to spot failure |
| gradients must be cut | the trunk never learns anything *for* the head |
| tied to one policy | a new checkpoint is a new feature space |
| architecture work | modifying the model, not wrapping it |

The last one is the point rather than the price.

## 12. What to do with the score

Everything up to here has been detection. What comes out is one number per frame, and a number on its own
changes nothing.

Section 2 named the two branches it feeds: **close the gap**, or **hand it over to teleop**.

Closing the gap means trying to recover the rollout. It was explored, and nothing survived:

| path | why it was cut |
|---|---|
| best-of-N steering | ACT is deterministic — measured 0.000% spread, so there are no candidates to rank |
| latent sampling | not exposed in any LeRobot release, fork, or `act-plus-plus` |
| rewind to last confident state | deterministic policy retraces the same path into the same failure |
| reset and retry | same, unless the scene changed |
| world model takeover | random search over ~280 dims, and it discards the trained policy exactly when the task is hardest |
| failure-type classification | too complex for a proof of concept |
| gradient refinement of the chunk | the one real correction mechanism found. Held in reserve, not built |

## 13. Handing it over

The other branch is the easy one. What this system does today is flag: the score crosses a line and it says
so. Piping that into a manual takeover is a small step — pause the policy, move to a neutral pose, give the
arm to an operator.

It was left out for time and hardware, and because the work here was the critic layer. The score is also
uncalibrated and carries no threshold; where that line sits belongs to whoever acts on it.

## 14. Next steps

| next step | why |
|---|---|
| harder tasks | one task and one failure mode today. Generality is unproven until the task list grows |
| train the policy and the head longer | the policy fails often by design, and the head saw a couple hundred rollouts |
| alarm harness | the score only prints. Wire it to an action — pause, neutral pose |
| teleop takeover harness | hand the arm to an operator once the alarm fires |
| the same head on a VLA | the design does not depend on ACT, only on a trunk whose features can be tapped |

## 15. Alternative routes

| route | what it changes |
|---|---|
| this head on a VLA | the design needs only a trunk whose features can be tapped. A VLA is where a multitask claim could actually be earned |
| a policy that samples | ACT is deterministic, so steering was dead on arrival. A VLA or diffusion policy with real sampling puts best-of-N back on the table |
| world model critic | judge a plan by rolling it forward in a learned model of the scene, rather than by reading the policy's own features |
| world model execution advising | sample candidate actions, roll each one forward, execute whichever looks best |
| detector trained with the policy | drop the freeze and the detach, so the trunk shapes its features for failure as well as for actions |
| a VLM watching from outside | no policy access at all. Slower and blind to internals, but it drops onto any robot |
| heuristic only | TCE and ACM thresholded, no learning at all. The floor everything else has to beat |
