# act-critic

ACT with a built-in failure head — one model, one forward pass, outputs both an action chunk and a live "am I failing now" score.

Most runtime failure detectors run as a second model alongside the policy. This one lives inside it.

I wanted to build it over a weekend.

See [docs/stack.md](docs/stack.md) for versions and hardware.
