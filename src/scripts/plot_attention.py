"""Average the critic head's attention over a dataset and draw it as two camera grids.

Answers what the head actually looks at. The weights are `aₖ`, ABMIL's softmax output,
averaged over every frame and every history offset.

    PYTHONPATH=$PWD/src .venv-lerobot/bin/python src/scripts/plot_attention.py \
      --dataset bjahoor/lift-cube-rollouts-10k

Token layout, verified by perturbing each input and watching the encoder's input change:
0 is the latent z, 1 the arm state, 2-50 the wrist camera, 51-99 the table camera. Each
camera is a 7x7 patch grid in row-major order, rows running from the top of the image down.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modeling.act_critic import CriticHead  # noqa: E402
from scripts.train_critic import CachedFrames, cache_path  # noqa: E402

TOP, LOW = {96, 97, 98}, {33, 34, 35}
C_TOP, C_LOW, C_LEG, C_NOT = "#00e676", "#ff2d55", "#1f6feb", "#7b3fe4"


def weights(head: CriticHead, data: CachedFrames, device: str, batch: int) -> np.ndarray:
    """Mean attention per token over every frame, as percentages."""
    total, n = torch.zeros(100, dtype=torch.float64), 0
    with torch.no_grad():
        for tokens, tce, acm, _ in DataLoader(data, batch_size=batch, num_workers=4):
            a = head(tokens.to(device), tce.to(device), acm.to(device))["attention"]
            total += a.double().mean(1).sum(0).cpu()
            n += a.shape[0]
    return (total / n).numpy() * 100, n


def draw(w: np.ndarray, frames: int, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    wrist, table = w[2:51].reshape(7, 7), w[51:100].reshape(7, 7)
    lo, hi = w.min(), w.max()
    # dark cells need light text, and the colormap is dark at the bottom
    ink = lambda v: "white" if v < (lo + hi) / 2 else "black"

    fig = plt.figure(figsize=(21, 6.6), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[3.0, 7, 7], wspace=0.40)

    # z and the arm state are not patches, so they sit apart with a gap between them
    ax = fig.add_subplot(gs[0])
    col = np.full((7, 1), np.nan)
    col[0, 0], col[2, 0] = w[0], w[1]
    ax.imshow(np.ma.masked_invalid(col), cmap="magma", vmin=lo, vmax=hi)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
    for row, k in ((0, 0), (2, 1)):
        ax.add_patch(Rectangle((-.5, row - .5), 1, 1, fill=False, ec="#333", lw=1.5))
        ax.text(0, row - .17, str(k), ha="center", va="center", fontsize=15, color=ink(w[k]))
        ax.text(0, row + .23, f"{w[k]:.2f}", ha="center", va="center", fontsize=13, color=ink(w[k]))
    ax.text(0.60, 0, "z", ha="left", va="center", fontsize=14, color=C_NOT, weight="bold")
    ax.text(0.60, 2, "arm state", ha="left", va="center", fontsize=14, color=C_NOT, weight="bold")
    ax.annotate("token number", xy=(-0.30, -0.20), xytext=(-3.7, -0.95), fontsize=14,
                color=C_LEG, va="center", ha="left",
                arrowprops=dict(arrowstyle="->", color=C_LEG, lw=2))
    ax.annotate("attention weight, %", xy=(-0.30, 0.26), xytext=(-3.9, 1.30), fontsize=14,
                color=C_LEG, va="center", ha="left",
                arrowprops=dict(arrowstyle="->", color=C_LEG, lw=2))

    def panel(a, grid, start, title):
        im = a.imshow(grid, cmap="magma", vmin=lo, vmax=hi)
        a.set_title(title, fontsize=17, pad=14)
        a.set_xticks([]); a.set_yticks([])
        for r in range(7):
            for c in range(7):
                k, v = start + r * 7 + c, grid[r, c]
                a.text(c, r - .17, str(k), ha="center", va="center", fontsize=13, color=ink(v))
                a.text(c, r + .23, f"{v:.2f}", ha="center", va="center", fontsize=11, color=ink(v))
                if k in TOP or k in LOW:
                    a.add_patch(Rectangle((c - .5, r - .5), 1, 1, fill=False,
                                          ec=C_TOP if k in TOP else C_LOW, lw=3.5))
        return im

    a0 = fig.add_subplot(gs[1]); im = panel(a0, wrist, 2, "wrist camera")
    a1 = fig.add_subplot(gs[2]); panel(a1, table, 51, "table camera")
    cb = fig.colorbar(im, ax=[a0, a1], fraction=0.02, pad=0.07)
    cb.set_label("attention weight  $a_k$  (%)", fontsize=14, labelpad=14)
    cb.ax.tick_params(labelsize=12)
    fig.suptitle(f"What the critic head looks at, averaged over {frames:,} unseen frames",
                 fontsize=19, y=1.05)
    fig.text(0.5, -0.06,
             "Each camera is a 7x7 grid of patches.\n"
             "Green outlines the three highest weights, red the three lowest.\n"
             "The two squares on the far left, z and the arm state, belong to neither camera.\n"
             "Mean pooling would put every token at 1.00.",
             ha="center", fontsize=14, color="#333")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--head", default="checkpoints/critic-abmil/critic.pt")
    p.add_argument("--dataset", default="bjahoor/lift-cube-rollouts-10k", help="unseen set")
    p.add_argument("--cache-dir", default="datasets/critic-cache")
    p.add_argument("--out", type=Path, default=Path("docs/images/attention_grid.png"))
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    saved = torch.load(args.head, map_location=args.device, weights_only=False)
    if args.dataset == saved["train_dataset"]:
        print(f"[warn] {args.dataset} is what the head trained on; weights will flatter it")
    head = CriticHead(pooling=saved["pooling"]).to(args.device).eval()
    head.load_state_dict(saved["head"])

    data = CachedFrames(cache_path(Path(args.cache_dir), args.dataset, saved["base"]))
    w, frames = weights(head, data, args.device, args.batch_size)
    order = np.argsort(-w)
    print(f"[data] {args.dataset}: {frames:,} frames")
    print("  top    ", [(int(k), round(float(w[k]), 3)) for k in order[:3]])
    print("  bottom ", [(int(k), round(float(w[k]), 3)) for k in order[-3:]])
    print(f"  median {np.median(w):.3f}   top ten hold {w[order[:10]].sum():.2f}%")
    draw(w, frames, args.out)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
