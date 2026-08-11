"""Move network for Stratego move selection (arXiv:2511.07312).

Encoder-only transformer that consumes the full board infostate and produces:
  - a categorical value estimate (win/loss/draw) from a learned value token, and
  - a move policy via a key-query matrix product head (Monroe 2024).

Architecture (tab:move-network-hyper):
  depth=8, dim=384, heads=8, ff=1536, pre-LN, learned absolute positional
  embeddings (init std=0.1), bidirectional (non-causal) self-attention.
  ~14.7M parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from stratego.constants import (
    MOVE_NET_DEPTH,
    MOVE_NET_DIM,
    MOVE_NET_FF,
    MOVE_NET_HEADS,
    MOVE_NET_POS_EMB_INIT_STD,
    NUM_INFOSTATE_CHANNELS,
    NUM_SQUARES,
)
from stratego.networks.transformer import (
    PositionalEmbedding,
    TransformerStack,
    ValueHead,
)

__all__ = ["KeyQueryPolicyHead", "MoveNetwork"]


class KeyQueryPolicyHead(nn.Module):
    """Key-query matrix product policy head (Monroe 2024).

    Single-head ``Q·K^T / sqrt(d)`` over the encoder's token sequence.  Unlike
    a symmetric dot-product, the query and key come from separate linear
    projections (``wq``, ``wk``), so the resulting logit matrix is asymmetric.

    Parameters
    ----------
    embedding_dim:
        Input token embedding dimension.
    policy_d_model:
        Internal dimension of the query/key projections.  Defaults to
        ``embedding_dim``.  The output logit matrix is always
        ``(B, S, S)`` regardless of this value.
    """

    def __init__(
        self,
        embedding_dim: int,
        policy_d_model: int | None = None,
    ) -> None:
        super().__init__()
        d = policy_d_model or embedding_dim
        self.policy_embedding = nn.Linear(embedding_dim, embedding_dim)
        self.wq = nn.Linear(embedding_dim, d, bias=False)
        self.wk = nn.Linear(embedding_dim, d, bias=False)
        self.scale: float = d ** 0.5
        nn.init.xavier_normal_(self.policy_embedding.weight)
        nn.init.xavier_normal_(self.wq.weight)
        nn.init.xavier_normal_(self.wk.weight)

    def forward(
        self,
        encoder_output: torch.Tensor,
        legal_move_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the (B, S*S) flattened policy logits.

        Parameters
        ----------
        encoder_output:
            Encoder token sequence of shape ``(B, S, embedding_dim)``.
        legal_move_mask:
            Optional boolean mask of shape ``(B, S, S)`` — ``True`` marks a
            legal (from, to) move.  Illegal entries are set to ``-1e10``.

        Returns
        -------
        torch.Tensor:
            Flattened logits of shape ``(B, S*S)``.
        """
        policy_tokens = F.mish(self.policy_embedding(encoder_output))
        queries = self.wq(policy_tokens)
        keys = self.wk(policy_tokens)
        logit_matrix = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale
        if legal_move_mask is not None:
            logit_matrix = logit_matrix.masked_fill(~legal_move_mask, -1e10)
        return logit_matrix.reshape(logit_matrix.shape[0], -1)


class MoveNetwork(nn.Module):
    """Encoder-only transformer for move selection. ~14.7M parameters.

    The network consumes a Stratego infostate ``(B, 488, 10, 10)`` and emits a
    value estimate from a learned value token plus a move policy from a
    key-query matrix product over the 100 board squares.

    Parameters
    ----------
    depth:
        Number of pre-LN transformer blocks.  Default: ``MOVE_NET_DEPTH`` (8).
    dim:
        Model / embedding dimension.  Default: ``MOVE_NET_DIM`` (384).
    heads:
        Number of attention heads.  Default: ``MOVE_NET_HEADS`` (8).
    ff:
        Feed-forward hidden dimension.  Default: ``MOVE_NET_FF`` (1536).
    pos_emb_std:
        Initial std for the learned absolute positional embedding.
        Default: ``MOVE_NET_POS_EMB_INIT_STD`` (0.1).
    num_squares:
        Number of board squares (encoder token count).  Default: ``NUM_SQUARES``
        (100).
    """

    def __init__(
        self,
        depth: int = MOVE_NET_DEPTH,
        dim: int = MOVE_NET_DIM,
        heads: int = MOVE_NET_HEADS,
        ff: int = MOVE_NET_FF,
        pos_emb_std: float = MOVE_NET_POS_EMB_INIT_STD,
        num_squares: int = NUM_SQUARES,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(NUM_INFOSTATE_CHANNELS, dim)
        self.pos_emb = PositionalEmbedding(
            max_len=num_squares + 1,
            embed_dim=dim,
            init_std=pos_emb_std,
            init="normal",
        )
        self.value_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.value_token, std=0.02)
        self.blocks = TransformerStack(
            depth=depth,
            embed_dim=dim,
            num_heads=heads,
            ff_dim=ff,
            dropout=0.0,
            causal=False,
        )
        self.value_head = ValueHead(dim)
        self.policy_head = KeyQueryPolicyHead(dim, dim)

    def forward(
        self,
        infostate: torch.Tensor,
        legal_move_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass over a Stratego infostate.

        Parameters
        ----------
        infostate:
            Board state of shape ``(B, 488, 10, 10)`` (channel-first) or
            already-flattened ``(B, 100, 488)``.
        legal_move_mask:
            Optional boolean mask of shape ``(B, 100, 100)`` forwarded to the
            policy head.  ``True`` marks a legal move.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]:
            ``(value_logits, policy_logits)`` where ``value_logits`` is
            ``(B, 3)`` (win/loss/draw) and ``policy_logits`` is ``(B, 10000)``.
        """
        batch_size = infostate.shape[0]
        # (B, C, H, W) -> (B, H*W, C) = (B, 100, 488); pass through if already flat.
        x = infostate.permute(0, 2, 3, 1).flatten(1, 2) if infostate.dim() == 4 else infostate
        x = self.input_proj(x)  # (B, 100, dim)
        val_tok = self.value_token.expand(batch_size, -1, -1)  # (B, 1, dim)
        x = torch.cat([val_tok, x], dim=1)  # (B, 101, dim)
        x = self.pos_emb(x)
        x = self.blocks(x)  # (B, 101, dim)
        value = self.value_head(x[:, 0])  # (B, 3)
        policy = self.policy_head(x[:, 1:], legal_move_mask=legal_move_mask)  # (B, 10000)
        return value, policy
