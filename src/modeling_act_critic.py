"""ACT with a critic head: the policy's action chunk and a failure score from one forward pass.

The trunk is LeRobot's ACT, frozen. The head reads the encoder output and never
touches the decoder, so the policy's behaviour is unchanged.

`CriticHead` is usable on its own against cached encoder output, which is how it
trains. `ACTWithCritic` is the deployed pair, and is what the rollout loop uses.
"""

from __future__ import annotations

from collections import deque

import torch
from torch import Tensor, nn

from lerobot.policies.act.modeling_act import ACTPolicy

# frames back from now, at 50 fps: now, 0.1 s, 0.2 s, 0.3 s
HISTORY_OFFSETS = (0, 5, 10, 15)

# the encoder emits 512-wide tokens. 512 gives the head 5.9M parameters against ~200
# episodes, which memorises; 128 gives 0.1M
HEAD_DIM = 128

# matches ACT's own dropout, rather than a number picked here
DROPOUT = 0.1


class GatedAttentionPool(nn.Module):
    """ABMIL gated attention pooling, Ilse et al. ICML 2018.

    Scores each token on its own and takes the weighted sum. The encoder's self-attention
    already related the tokens to each other, so re-relating them here would pay twice for
    the frozen trunk's work.
    """

    def __init__(self, dim: int = HEAD_DIM):
        super().__init__()
        self.value = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.score = nn.Linear(dim, 1)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        """(B, N, D) -> pooled (B, D) and the per-token weights (B, N).

        The weights are returned so they can be drawn over the camera image. A head that
        attends to the gripper and cube found the intended signal; one that attends to a
        corner of the table found a shortcut.
        """
        gated = torch.tanh(self.value(tokens)) * torch.sigmoid(self.gate(tokens))
        weights = torch.softmax(self.score(gated).squeeze(-1), dim=-1)
        return torch.einsum("bn,bnd->bd", weights, tokens), weights


class MeanPool(nn.Module):
    """The control. If this ties the gated version, attention did not earn its place."""

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        b, n, _ = tokens.shape
        return tokens.mean(dim=1), tokens.new_full((b, n), 1.0 / n)


class CriticHead(nn.Module):
    """Encoder tokens over a short history, plus TCE and ACM, to one failure score.

    TCE and ACM arrive already normalized — they are raw joint units at source, where the
    arm's radians would drown the fingers' metres.
    """

    def __init__(self, token_dim: int = 512, dim: int = HEAD_DIM, pooling: str = "abmil"):
        super().__init__()
        self.project = nn.Linear(token_dim, dim)
        self.dropout = nn.Dropout(DROPOUT)
        self.pool = GatedAttentionPool(dim) if pooling == "abmil" else MeanPool()
        # one pooled vector per history frame, kept separate so the ordering survives,
        # plus the two scalars, which have nowhere to attend and so join at the output
        self.score = nn.Linear(dim * len(HISTORY_OFFSETS) + 2, 1)

    def forward(self, tokens: Tensor, tce: Tensor, acm: Tensor) -> dict[str, Tensor]:
        """tokens (B, T, N, 512) with T history frames, tce and acm (B, 1) each.

        Returns the 0-1 score, the raw value the loss needs for numerical stability, and
        the attention weights.
        """
        b, t, n, _ = tokens.shape
        # the cache is float16 to halve what crosses PCIe; a no-op when already float32
        x = self.dropout(self.project(tokens.float())).reshape(b * t, n, -1)
        pooled, weights = self.pool(x)
        logit = self.score(torch.cat([pooled.reshape(b, -1), tce, acm], dim=-1))
        return {
            "failure_score": torch.sigmoid(logit).squeeze(-1),
            "logit": logit.squeeze(-1),
            "attention": weights.reshape(b, t, n),
        }


class ACTWithCritic(ACTPolicy):
    """ACT, frozen, with the critic head reading its encoder output.

    The trunk is never in the head's gradient path: its parameters are frozen and the
    tokens are detached at the tap. `select_action` and `predict_action_chunk` are
    inherited untouched, so the action chunk is bit-identical to the base checkpoint's.
    """

    def __init__(self, config, pooling: str = "abmil", **kwargs):
        super().__init__(config, **kwargs)
        self.critic = CriticHead(config.dim_model, pooling=pooling)
        self.freeze_trunk()
        # the encoder's output is computed once and handed only to the decoder, so it is
        # taken with a hook rather than by reimplementing ACT.forward
        self._tokens: Tensor | None = None
        self._fresh = False
        self.model.encoder.register_forward_hook(self._capture)
        self._history: deque[Tensor] = deque(maxlen=max(HISTORY_OFFSETS) + 1)

    def freeze_trunk(self) -> None:
        for name, p in self.named_parameters():
            if not name.startswith("critic."):
                p.requires_grad_(False)

    def _capture(self, _module, _inputs, output: Tensor) -> None:
        # ACT works in (sequence, batch, dim); everything downstream here is batch-first
        self._tokens = output.detach().transpose(0, 1)
        self._fresh = True

    def reset(self) -> None:
        super().reset()
        # ACTPolicy.__init__ calls reset() before this subclass has built its buffers
        if hasattr(self, "_history"):
            self._history.clear()
            self._fresh = False

    @torch.no_grad()
    def critic_score(self, batch: dict[str, Tensor], tce: Tensor, acm: Tensor,
                     chunk: Tensor | None = None) -> dict[str, Tensor] | None:
        """Score the current frame, and return the action chunk it was scored from.

        The encoder must run on every frame. `select_action` only runs it every
        `n_action_steps` — 10 for this checkpoint — so relying on whatever the hook last
        caught gives a history of 2 distinct frames where training used 4, silently.

        A caller that already ran `predict_action_chunk` this frame — to compute TCE and ACM
        from the chunk — passes it in and the trunk runs once rather than twice. Passing a
        stale chunk raises rather than quietly scoring duplicated history.

        Returns None until the history fills, 0.3 s into an episode. Padding with copies of
        the first frame would invent stillness exactly where the approach is decided.
        """
        if chunk is None:
            chunk = self.predict_action_chunk(batch)
        elif not self._fresh:
            raise RuntimeError("chunk passed but the encoder has not run since the last "
                               "critic_score; call predict_action_chunk on this frame first")
        self._fresh = False
        self._history.append(self._tokens)
        if len(self._history) <= max(HISTORY_OFFSETS):
            return None
        tokens = torch.stack([self._history[-1 - k] for k in HISTORY_OFFSETS], dim=1)
        return {**self.critic(tokens, tce, acm), "action_chunk": chunk}
