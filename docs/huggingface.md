# Hugging Face

Every Hub repo this project reads or writes.

**Datasets**

| Repo | What it is |
|---|---|
| [bjahoor/lift-cube-franka-v2](https://huggingface.co/datasets/bjahoor/lift-cube-franka-v2) | 50 scripted demos, the ACT training set |
| [bjahoor/lift-cube-franka](https://huggingface.co/datasets/bjahoor/lift-cube-franka) | First demo set, discarded |
| [bjahoor/lift-cube-rollouts-20k-200](https://huggingface.co/datasets/bjahoor/lift-cube-rollouts-20k-200) | 200 rollouts of the 20k policy, critic training set |
| [bjahoor/lift-cube-rollouts-20k-200-chunks](https://huggingface.co/datasets/bjahoor/lift-cube-rollouts-20k-200-chunks) | Cached scene tokens for that set |
| [bjahoor/lift-cube-rollouts-20k](https://huggingface.co/datasets/bjahoor/lift-cube-rollouts-20k) | Earlier 100 rollouts, tuning / early stopping |
| [bjahoor/lift-cube-rollouts-10k](https://huggingface.co/datasets/bjahoor/lift-cube-rollouts-10k) | Rollouts of the 10k policy, held-out transfer test |
| [bjahoor/lift-cube-rollouts-10k-chunks](https://huggingface.co/datasets/bjahoor/lift-cube-rollouts-10k-chunks) | Cached scene tokens for that set |

**Models**

| Repo | What it is |
|---|---|
| [bjahoor/act-lift-cube-franka-v2-10k](https://huggingface.co/bjahoor/act-lift-cube-franka-v2-10k) | ACT checkpoint, 10k steps |
| [bjahoor/act-lift-cube-franka-v2-20k](https://huggingface.co/bjahoor/act-lift-cube-franka-v2-20k) | ACT checkpoint, 20k steps, the frozen critic trunk |
| [bjahoor/act-lift-cube-franka-v2-30k](https://huggingface.co/bjahoor/act-lift-cube-franka-v2-30k) | ACT checkpoint, 30k steps |
| [bjahoor/act-lift-cube-franka-v2-40k](https://huggingface.co/bjahoor/act-lift-cube-franka-v2-40k) | ACT checkpoint, 40k steps |
| [bjahoor/act-lift-cube-franka-v2](https://huggingface.co/bjahoor/act-lift-cube-franka-v2) | ACT checkpoint, 50k steps, final |
| [bjahoor/act-critic-head](https://huggingface.co/bjahoor/act-critic-head) | The trained critic head. What the scripts download by default |
