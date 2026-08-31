"""Score a trained critic head offline: threshold sweep, earliness, and the baselines.

`eval_critic.py` is the live on-screen demo. This is the numbers.

An alarm fires when the score stays above a threshold for `--hold` consecutive frames, so a
single noisy frame is not a detection. Every baseline is put through the same rule, or the
comparison is not one.

    PYTHONPATH=$PWD/src .venv-lerobot/bin/python src/scripts/measure_critic.py \
      --dataset bjahoor/lift-cube-rollouts-10k
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from modeling_act_critic import HISTORY_OFFSETS, CriticHead
from train_critic import CachedFrames, average_precision, build_cache, cache_path

FPS = 50

# the approach is decided at roughly 1.0-1.8 s, and a success is over by ~3 s. An alarm
# after that is aftermath, not warning
USEFUL_BY_S = 3.0


def episode_scores(head: CriticHead, dataset: CachedFrames, device: str, scale: dict,
                   batch: int = 512) -> list[np.ndarray]:
    """One score per frame, grouped by episode, in frame order."""
    head.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(len(dataset.files)):
            m = dataset.meta[i]
            frames = range(max(HISTORY_OFFSETS), len(m["tce"]))
            scores = []
            for lo in range(0, len(frames), batch):
                window = list(frames)[lo:lo + batch]
                tokens = torch.from_numpy(
                    np.stack([np.stack([dataset._tokens(i)[t - o] for o in HISTORY_OFFSETS]) for t in window])
                ).to(device)
                tce = torch.tensor([[(m["tce"][t] - scale["tce_mean"]) / scale["tce_std"]] for t in window],
                                   dtype=torch.float32, device=device)
                acm = torch.tensor([[(m["acm"][t] - scale["acm_mean"]) / scale["acm_std"]] for t in window],
                                   dtype=torch.float32, device=device)
                scores.append(head(tokens, tce, acm)["failure_score"].float().cpu().numpy())
            out.append(np.concatenate(scores))
    return out


def alarm_frame(scores: np.ndarray, threshold: float, hold: int, offset: int) -> int | None:
    """First frame where the score has been above the threshold for `hold` frames running."""
    above = scores > threshold
    if len(above) < hold:
        return None
    run = np.convolve(above.astype(int), np.ones(hold, dtype=int), mode="valid")
    hit = np.flatnonzero(run == hold)
    return None if len(hit) == 0 else int(hit[0]) + hold - 1 + offset


def sweep(per_episode: list[np.ndarray], labels: np.ndarray, hold: int, offset: int) -> list[dict]:
    rows = []
    for threshold in np.arange(0.05, 1.0, 0.05):
        fired = [alarm_frame(s, threshold, hold, offset) for s in per_episode]
        tp = [f for f, y in zip(fired, labels) if y == 1 and f is not None]
        fp = sum(1 for f, y in zip(fired, labels) if y == 0 and f is not None)
        n_fail = int(labels.sum())
        precision = len(tp) / max(len(tp) + fp, 1)
        rows.append({
            "threshold": float(threshold),
            "recall": len(tp) / max(n_fail, 1),
            "precision": precision,
            "false_alarms": fp / max(int((labels == 0).sum()), 1),
            "median_alarm": float(np.median(tp)) / FPS if tp else float("nan"),
            "caught_early": sum(1 for f in tp if f / FPS <= USEFUL_BY_S) / max(n_fail, 1),
        })
    return rows


def report(name: str, rows: list[dict]) -> None:
    print(f"\n{name}")
    print(f"  {'thresh':>7}{'recall':>9}{'prec':>8}{'false':>8}{'alarm s':>9}{'early':>8}")
    for r in rows:
        med = "  -" if np.isnan(r["median_alarm"]) else f"{r['median_alarm']:.2f}"
        print(f"  {r['threshold']:>7.2f}{r['recall']:>9.2f}{r['precision']:>8.2f}"
              f"{r['false_alarms']:>8.2f}{med:>9}{r['caught_early']:>8.2f}")
    best = max(rows, key=lambda r: 2 * r["precision"] * r["recall"] / max(r["precision"] + r["recall"], 1e-9))
    med = "n/a" if np.isnan(best["median_alarm"]) else f"{best['median_alarm']:.2f} s"
    print(f"  best F1 at {best['threshold']:.2f}: recall {best['recall']:.2f} "
          f"precision {best['precision']:.2f}, median alarm at {med}, "
          f"caught early {best['caught_early']:.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="bjahoor/lift-cube-rollouts-10k", help="held-out set to score")
    p.add_argument("--head", default="checkpoints/critic-abmil/critic.pt")
    # default is the trunk the head was trained against: unseen episodes, same perception.
    # pointing this at the checkpoint that generated the rollouts instead asks the harder
    # question, whether the head survives a different policy's feature space
    p.add_argument("--checkpoint", default=None, help="trunk to encode with (default: the head's own)")
    p.add_argument("--cache-dir", default="datasets/critic-cache")
    p.add_argument("--hold", type=int, default=5, help="consecutive frames above threshold to alarm")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    saved = torch.load(args.head, map_location=args.device, weights_only=False)
    if tuple(saved["history_offsets"]) != HISTORY_OFFSETS:
        raise SystemExit(f"head was trained with offsets {saved['history_offsets']}, code has {HISTORY_OFFSETS}")
    head = CriticHead(pooling=saved["pooling"]).to(args.device)
    head.load_state_dict(saved["head"])
    print(f"[head] {args.head}  pooling {saved['pooling']}  trained on {saved['train_dataset']}")
    if args.dataset == saved["train_dataset"]:
        print("[head] WARNING: scoring the set it was trained on")

    trunk = args.checkpoint or saved["base"]
    if trunk != saved["base"]:
        print(f"[head] encoding with {trunk}, not the trunk the head was trained on")
    cache = build_cache(args.dataset, trunk,
                        cache_path(Path(args.cache_dir), args.dataset, trunk), 32, args.device)
    data = CachedFrames(cache)
    labels = np.array([float(m["failure"]) for m in data.meta])
    offset = max(HISTORY_OFFSETS)
    print(f"[data] {args.dataset}: {len(data.files)} episodes, "
          f"{int(labels.sum())} failures, {len(data)} scored frames "
          f"({len(data) / FPS:.0f} s of robot time)")

    scores = episode_scores(head, data, args.device, saved["scale"])
    ep_score = np.array([s.mean() for s in scores])
    print(f"\n[AP] head episode AP {average_precision(labels, ep_score):.4f}")

    tce = [m["tce"][offset:] for m in data.meta]
    print(f"[AP] TCE alone       {average_precision(labels, np.array([t.mean() for t in tce])):.4f}")
    print(f"[AP] always-fail     {labels.mean():.4f}")

    report(f"HEAD  (hold {args.hold} frames)", sweep(scores, labels, args.hold, offset))
    # the same alarm rule on the free signal, or the comparison is not one. TCE is
    # standardized to 0-1 by rank so the same thresholds mean something
    ranked = [np.argsort(np.argsort(np.concatenate(tce))).astype(float)]
    flat = ranked[0] / len(ranked[0])
    split, tce_ranked = 0, []
    for t in tce:
        tce_ranked.append(flat[split:split + len(t)])
        split += len(t)
    report(f"TCE ALONE  (hold {args.hold} frames)", sweep(tce_ranked, labels, args.hold, offset))

    print(f"\nalarm s = when the alarm first fires on caught failures. early = fired within "
          f"{USEFUL_BY_S:.0f} s, before a success would have finished.")


if __name__ == "__main__":
    main()
