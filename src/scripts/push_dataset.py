"""Push a recorded LeRobot dataset to the HuggingFace Hub.

No Isaac imports — run with the venv python directly:

    .venv-lerobot/bin/python src/scripts/push_dataset.py --repo_id bjahoor/lift-cube-franka
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    parser = argparse.ArgumentParser(description="Push a recorded dataset to the HuggingFace Hub.")
    parser.add_argument("--repo_id", type=str, required=True, help="Target hub repo, <user>/<name>.")
    parser.add_argument("--root", type=str, default="datasets/lift-cube-franka", help="Local dataset directory.")
    parser.add_argument("--private", action="store_true", default=False, help="Create the repo private.")
    parser.add_argument("--force", action="store_true", default=False, help="Overwrite an existing hub repo.")
    args = parser.parse_args()

    # a missing or wrong root makes LeRobotDataset download from the hub into it, so check first
    root = Path(args.root)
    if not (root / "meta" / "info.json").exists() or not list(root.glob("data/**/*.parquet")):
        sys.exit(f"{root} is not a recorded dataset — no meta/info.json or data parquet files.")

    if not args.force and HfApi().repo_exists(args.repo_id, repo_type="dataset"):
        sys.exit(f"{args.repo_id} already exists. Pass --force to overwrite it.")

    dataset = LeRobotDataset(args.repo_id, root=root)

    # info.json is written up front, so compare it against what is actually on disk
    on_disk = len(set(dataset.hf_dataset.unique("episode_index")))
    if on_disk != dataset.num_episodes:
        sys.exit(f"truncated: metadata claims {dataset.num_episodes} episodes, {on_disk} on disk.")

    print(f"{root}: {dataset.num_episodes} episodes, {dataset.num_frames} frames, {dataset.fps} fps")
    dataset.push_to_hub(private=args.private, tags=["isaac-lab", "franka", "act"])
    print(f"pushed to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
