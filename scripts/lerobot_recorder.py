"""Buffer simulator frames and write successful episodes as a LeRobot dataset.

No Isaac Lab imports — takes plain arrays or torch tensors so it can be tested
without a simulator.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

TASK = "Pick up the cube and lift it to the target position."

# why an episode ended, stored per frame in rollout mode
TIMEOUT, DROPPED, SUCCESS = 0, 1, 2


def _to_numpy(value) -> np.ndarray:
    """Accept a torch tensor (CPU or GPU) or a numpy array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class EpisodeRecorder:
    """Collects frames per environment and saves only the episodes that succeed."""

    def __init__(
        self,
        repo_id: str,
        root: str,
        num_envs: int,
        state_dim: int,
        image_shape: tuple[int, int, int],
        fps: int = 50,
        overwrite: bool = False,
        keep_failures: bool = False,
    ):
        root_path = Path(root)
        if root_path.exists():
            if not overwrite:
                raise FileExistsError(f"{root_path} already exists. Pass overwrite=True to replace it.")
            shutil.rmtree(root_path)

        self.state_dim = state_dim
        self.image_shape = image_shape
        self.keep_failures = keep_failures
        # lerobot rejects a frame carrying a field the dataset did not declare, so these
        # are declared only when they will be written. the cube position is what makes the
        # moment of failure recoverable afterwards; without it the label is episode-wide
        rollout_features = {
            "failure": {"dtype": "float32", "shape": (1,), "names": None},
            "object_pos": {"dtype": "float32", "shape": (3,), "names": None},
            "termination": {"dtype": "float32", "shape": (1,), "names": None},
        } if keep_failures else {}
        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=root_path,
            fps=fps,
            # writer threads keep PNG writes off the simulation loop
            image_writer_threads=4,
            features={
                "observation.images.wrist": {"dtype": "video", "shape": image_shape, "names": ["height", "width", "channels"]},
                "observation.images.table": {"dtype": "video", "shape": image_shape, "names": ["height", "width", "channels"]},
                "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": None},
                "action": {"dtype": "float32", "shape": (state_dim,), "names": None},
                **rollout_features,
            },
        )
        self.buffers: list[list[dict]] = [[] for _ in range(num_envs)]
        self.saved = 0
        self._closed = False

    def _image(self, value) -> np.ndarray:
        image = _to_numpy(value)
        if image.dtype != np.uint8:
            raise TypeError(f"images must be uint8, got {image.dtype}. Set normalize=False on the image observation.")
        if image.shape != self.image_shape:
            raise ValueError(f"expected image shape {self.image_shape}, got {image.shape}")
        return image

    def _vector(self, value, name: str, dim: int | None = None) -> np.ndarray:
        dim = self.state_dim if dim is None else dim
        vector = _to_numpy(value).astype(np.float32)
        if vector.shape != (dim,):
            raise ValueError(f"expected {name} shape {(dim,)}, got {vector.shape}")
        return vector

    def add(self, env_id: int, wrist, table, state, action, object_pos=None) -> None:
        """Buffer one frame. Nothing touches disk until the episode ends."""
        if not 0 <= env_id < len(self.buffers):
            raise IndexError(f"env_id {env_id} out of range for {len(self.buffers)} environments")
        frame = {
            "observation.images.wrist": self._image(wrist),
            "observation.images.table": self._image(table),
            "observation.state": self._vector(state, "state"),
            "action": self._vector(action, "action"),
            "task": TASK,
        }
        if self.keep_failures:
            frame["object_pos"] = self._vector(object_pos, "object_pos", 3)
        self.buffers[env_id].append(frame)

    def drop(self, env_id: int) -> None:
        """Discard a buffered episode without writing it."""
        self.buffers[env_id] = []

    def finish(self, env_id: int, success: bool, reason: int = TIMEOUT) -> None:
        """Write the buffered episode, then clear the buffer. Failures are kept only in
        rollout mode, where they are the point."""
        frames = self.buffers[env_id]
        if not frames or (not success and not self.keep_failures):
            self.drop(env_id)
            return
        # 1 is failure, matching what the head predicts. per episode, but lerobot stores per frame
        value = np.array([float(not success)], dtype=np.float32)
        termination = np.array([float(reason)], dtype=np.float32)
        try:
            for frame in frames:
                if self.keep_failures:
                    frame["failure"] = value
                    frame["termination"] = termination
                self.dataset.add_frame(frame)
            # forking to encode video from a live CUDA process can hang
            self.dataset.save_episode(parallel_encoding=False)
            self.saved += 1
        except Exception:
            # leave no partial frames behind, or the next episode appends onto them
            self.dataset.clear_episode_buffer()
            raise
        finally:
            self.drop(env_id)

    def close(self) -> None:
        # called explicitly after the loop and again via atexit, so make it idempotent
        if self._closed:
            return
        self._closed = True
        self.dataset.finalize()
