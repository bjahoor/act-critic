"""Train the critic head on recorded rollouts.

Two passes. The first runs the frozen trunk over every frame once and caches the encoder
output, because the trunk never changes and decoding 200 videos per epoch would dominate
the run. The second trains the head on that cache, where ACT is not present at all.

    PYTHONPATH=$PWD/src .venv-lerobot/bin/python src/scripts/train_critic.py \
      --dataset bjahoor/lift-cube-rollouts-20k-200 \
      --tune-dataset bjahoor/lift-cube-rollouts-20k
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from huggingface_hub import snapshot_download
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset

from modeling_act_critic import HEAD_DIM, HISTORY_OFFSETS, ACTWithCritic, CriticHead

TASK = "Pick up the cube and lift it to the target position."

# the chunk from 10 steps ago overlaps this one by 90. comparing across a real replan gap
# rather than adjacent frames, where the difference is mostly noise
TCE_GAP = 10

SEED = 0


def compute_tce_acm(chunks: np.ndarray, std: np.ndarray, mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(T, 100, 9) raw joint targets -> per-frame TCE and ACM.

    Normalized with the checkpoint's own action statistics first. Raw, the arm's radians
    dwarf the fingers' metres and ACM would be ~99.9% arm.
    """
    c = (chunks - mean) / std
    acm = np.sqrt((c**2).mean(axis=(1, 2)))
    tce = np.zeros(len(c), dtype=np.float32)
    if len(c) > TCE_GAP:
        tce[TCE_GAP:] = ((c[TCE_GAP:, :-TCE_GAP] - c[:-TCE_GAP, TCE_GAP:]) ** 2).mean(axis=(1, 2))
    return tce.astype(np.float32), acm.astype(np.float32)


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the precision-recall curve. Written out rather than pulling in sklearn,
    which would have to install into a venv sharing Isaac's site-packages."""
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], labels[order]
    tp, fp = np.cumsum(y), np.cumsum(1 - y)
    # one point per distinct score, or a model that outputs a constant scores above its
    # base rate purely from how ties happen to sort
    last = np.r_[np.diff(s) != 0, True]
    tp, fp = tp[last], fp[last]
    recall = tp / max(y.sum(), 1)
    return float((np.diff(np.r_[0.0, recall]) * (tp / (tp + fp))).sum())


def cache_path(root: Path, repo_id: str, checkpoint: str) -> Path:
    """Keyed by both, so the transfer set cannot silently reuse the training set's cache."""
    return root / f"{repo_id.split('/')[-1]}__{checkpoint.split('/')[-1]}"


def build_cache(repo_id: str, checkpoint: str, out: Path, batch_size: int, device: str) -> Path:
    """Run the frozen trunk over every frame once and write one episode per file."""
    out.mkdir(parents=True, exist_ok=True)
    done = {int(p.stem.split("_")[1]) for p in out.glob("episode_*_meta.npz")}

    dataset = LeRobotDataset(repo_id)
    n_episodes = dataset.meta.total_episodes
    if len(done) >= n_episodes:
        print(f"[cache] {out.name} complete, {n_episodes} episodes")
        return out

    chunk_dir = Path(snapshot_download(f"{repo_id}-chunks", repo_type="dataset"))
    policy = ACTWithCritic.from_pretrained(checkpoint).to(device).eval()
    pre, _ = make_pre_post_processors(policy.config, pretrained_path=checkpoint)
    # the action statistics live in the checkpoint's normalizer, not in the dataset. the
    # policy's own stats, so TCE means what the policy experienced
    norm = load_file(
        Path(snapshot_download(checkpoint)) / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    )
    mean, std = norm["action.mean"].numpy(), norm["action.std"].numpy()

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    buffers: dict[int, tuple[list, float]] = {}
    with torch.no_grad():
        for batch in loader:
            episodes = batch["episode_index"].numpy()
            # a crash mid-cache would otherwise re-run the trunk over everything
            if set(episodes.tolist()) <= done:
                continue
            obs = {
                "observation.images.wrist": batch["observation.images.wrist"].to(device),
                "observation.images.table": batch["observation.images.table"].to(device),
                "observation.state": batch["observation.state"].to(device),
                "task": [TASK] * len(episodes),
            }
            policy.predict_action_chunk(pre(obs))
            tokens = policy._tokens.to(torch.float16).cpu().numpy()
            failures = batch["failure"].numpy().reshape(len(episodes))
            for i, ep in enumerate(episodes):
                buffers.setdefault(int(ep), ([], failures[i]))[0].append(tokens[i])
            # episodes arrive in order, so anything but the newest is finished
            for ep in sorted(buffers)[:-1]:
                write_episode(out, ep, *buffers.pop(ep), chunk_dir, mean, std)
                done.add(ep)
                print(f"[cache] {out.name} {len(done)}/{n_episodes}", end="\r", flush=True)
    for ep in sorted(buffers):
        write_episode(out, ep, *buffers.pop(ep), chunk_dir, mean, std)
    print(f"\n[cache] {out.name} done")
    return out


def write_episode(out: Path, ep: int, tokens: list[np.ndarray], failure: float,
                  chunk_dir: Path, mean, std) -> None:
    chunks = np.load(chunk_dir / f"episode_{ep:06d}.npy")
    tok = np.stack(tokens)
    # the recorder saved one chunk per recorded frame, so a mismatch means the two sources
    # are misaligned and every TCE would be off by a frame
    assert len(chunks) == len(tok), f"episode {ep}: {len(chunks)} chunks, {len(tok)} frames"
    tce, acm = compute_tce_acm(chunks, std, mean)
    # a bare .npy so training can memory-map it. mmap_mode is silently ignored on .npz,
    # which loads the whole episode into every dataloader worker instead
    np.save(out / f"episode_{ep:06d}_tokens.npy", tok)
    np.savez(out / f"episode_{ep:06d}_meta.npz", tce=tce, acm=acm, failure=np.float32(failure))


class CachedFrames(Dataset):
    """Frames with a full history window. The first 15 of each episode have none."""

    def __init__(self, cache: Path):
        self.files = sorted(cache.glob("episode_*_tokens.npy"))
        if not self.files:
            raise FileNotFoundError(f"no cached episodes in {cache}")
        self.index: list[tuple[int, int]] = []
        self.meta: list[dict] = []
        episode = []
        for i, f in enumerate(self.files):
            with np.load(str(f).replace("_tokens.npy", "_meta.npz")) as d:
                self.meta.append({k: d[k] for k in ("tce", "acm", "failure")})
            for t in range(max(HISTORY_OFFSETS), len(self.meta[i]["tce"])):
                self.index.append((i, t))
                episode.append(i)
        self.episode = np.array(episode)
        self._maps: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.index)

    def _tokens(self, i: int) -> np.ndarray:
        # memory-mapped, so pages are shared between workers and reclaimable under pressure
        if i not in self._maps:
            self._maps[i] = np.load(self.files[i], mmap_mode="r")
        return self._maps[i]

    def __getitem__(self, k: int):
        i, t = self.index[k]
        m = self.meta[i]
        tokens = np.stack([self._tokens(i)[t - o] for o in HISTORY_OFFSETS])
        # kept float16 all the way to the device, halving what crosses PCIe. CriticHead
        # casts after the transfer
        return (torch.from_numpy(tokens),
                torch.tensor([m["tce"][t]]), torch.tensor([m["acm"][t]]),
                torch.tensor(float(m["failure"])))


def evaluate(head: CriticHead, dataset: CachedFrames, loader: DataLoader, device: str, norm) -> dict:
    """Episode AP is what early stopping watches.

    Frame AP is nearly saturated by episode length — failures run to the 500-step giveup
    and successes stop at ~150, so frame index alone scores 0.913 on it. It is reported
    anyway, next to the baseline it has to clear.
    """
    head.eval()
    scores, labels = [], []
    with torch.no_grad():
        for tokens, tce, acm, y in loader:
            out = head(tokens.to(device, non_blocking=True), *norm(tce.to(device), acm.to(device)))
            scores.append(out["failure_score"].float().cpu().numpy())
            labels.append(y.numpy())
    head.train()
    scores, labels = np.concatenate(scores), np.concatenate(labels)
    ep_score = np.array([scores[dataset.episode == e].mean() for e in range(len(dataset.files))])
    ep_label = np.array([float(m["failure"]) for m in dataset.meta])
    return {"episode_ap": average_precision(ep_label, ep_score),
            "frame_ap": average_precision(labels, scores)}


def baselines(dataset: CachedFrames) -> dict:
    """What the head has to beat, from the cache alone. No model involved."""
    frame_label = np.array([float(dataset.meta[i]["failure"]) for i, _ in dataset.index])
    step = np.array([t for _, t in dataset.index], dtype=float)
    tce = np.array([dataset.meta[i]["tce"][t] for i, t in dataset.index], dtype=float)
    ep_label = np.array([float(m["failure"]) for m in dataset.meta])
    ep_tce = np.array([tce[dataset.episode == e].mean() for e in range(len(dataset.files))])
    return {"always-fail, episode AP": float(ep_label.mean()),
            "always-fail, frame AP": float(frame_label.mean()),
            "frame index, frame AP": average_precision(frame_label, step),
            "TCE alone, episode AP": average_precision(ep_label, ep_tce)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="bjahoor/lift-cube-rollouts-20k-200")
    p.add_argument("--tune-dataset", default="bjahoor/lift-cube-rollouts-20k")
    p.add_argument("--checkpoint", default="bjahoor/act-lift-cube-franka-v2-20k")
    p.add_argument("--pooling", default="abmil", choices=["abmil", "mean"])
    p.add_argument("--cache-dir", default="datasets/critic-cache")
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    # Ampere, and the head is 0.1M parameters, so this is free
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    root = Path(args.cache_dir)
    train_cache = build_cache(args.dataset, args.checkpoint,
                              cache_path(root, args.dataset, args.checkpoint), 32, args.device)
    tune_cache = build_cache(args.tune_dataset, args.checkpoint,
                             cache_path(root, args.tune_dataset, args.checkpoint), 32, args.device)

    train_set, tune_set = CachedFrames(train_cache), CachedFrames(tune_cache)
    print(f"[data] train {len(train_set)} frames / {len(train_set.files)} episodes, "
          f"tune {len(tune_set)} frames / {len(tune_set.files)} episodes")
    for name, value in baselines(tune_set).items():
        print(f"[baseline] {name:26s} {value:.4f}")

    # TCE and ACM are on wildly different scales to each other and to the tokens, so they
    # are standardized on the training set and the scaler travels with the head
    tce = np.array([train_set.meta[i]["tce"][t] for i, t in train_set.index])
    acm = np.array([train_set.meta[i]["acm"][t] for i, t in train_set.index])
    scale = {"tce_mean": float(tce.mean()), "tce_std": float(tce.std() + 1e-8),
             "acm_mean": float(acm.mean()), "acm_std": float(acm.std() + 1e-8)}

    def norm(a, b):
        return ((a - scale["tce_mean"]) / scale["tce_std"], (b - scale["acm_mean"]) / scale["acm_std"])

    head = CriticHead(pooling=args.pooling).to(args.device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    common = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **common)
    tune_loader = DataLoader(tune_set, batch_size=args.batch_size, **common)

    out = Path(args.out) / f"critic-{args.pooling}"
    out.mkdir(parents=True, exist_ok=True)
    best, waited = -1.0, 0
    for epoch in range(args.epochs):
        total, seen = 0.0, 0
        for tokens, t, a, y in train_loader:
            pred = head(tokens.to(args.device, non_blocking=True),
                        *norm(t.to(args.device), a.to(args.device)))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred["logit"], y.to(args.device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, seen = total + loss.item() * len(y), seen + len(y)
        m = evaluate(head, tune_set, tune_loader, args.device, norm)
        print(f"[train] epoch {epoch} loss {total / seen:.4f} "
              f"episode AP {m['episode_ap']:.4f}  frame AP {m['frame_ap']:.4f}")
        if m["episode_ap"] > best:
            best, waited = m["episode_ap"], 0
            torch.save({"head": head.state_dict(), "scale": scale, "pooling": args.pooling,
                        "base": args.checkpoint, "history_offsets": HISTORY_OFFSETS,
                        "head_dim": HEAD_DIM, "train_dataset": args.dataset}, out / "critic.pt")
        else:
            waited += 1
            if waited >= args.patience:
                print(f"[train] no improvement in {args.patience} epochs, stopping")
                break
    print(f"[train] best episode AP {best:.4f}, saved to {out / 'critic.pt'}")


if __name__ == "__main__":
    main()
