"""Decoder-only transformer for autoregressive setup generation (arXiv:2511.07312).

The setup network generates a player's 40-piece setup by autoregressively placing
pieces onto squares in row-major order. At each step it predicts:

  - the game outcome (win/loss/draw) conditional on the partial setup so far,
  - the conditional entropy of the outcome, and
  - a per-position distribution over the next piece type.

Architecture (tab:setup-network-hyper):
  depth=4, dim=512, heads=8, ff=2048, pre-LN, learned absolute positional
  embeddings (init std=0.1), causal self-attention. ~12.6M parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from stratego.constants import (
    SETUP_NET_DEPTH,
    SETUP_NET_DIM,
    SETUP_NET_FF,
    SETUP_NET_HEADS,
    SETUP_NET_POS_EMB_INIT_STD,
    TOTAL_PIECES,
)
from stratego.networks.transformer import (
    EntropyHead,
    PieceDistributionHead,
    PositionalEmbedding,
    TransformerStack,
    ValueHead,
)
from stratego.types import NUM_PIECE_TYPES

__all__ = ["SetupNetwork"]


class SetupNetwork(nn.Module):
    """Decoder-only transformer for autoregressive setup generation.

    Generates setups by autoregressively placing 40 pieces onto squares in
    row-major order. Outputs:

      - outcome logits (win/loss/draw) — ``(B, 3)``,
      - conditional entropy estimate — ``(B,)``,
      - next-piece distribution logits — ``(B, S, num_piece_types)``.

    Parameters
    ----------
    depth:
        Number of pre-LN transformer blocks. Default: ``SETUP_NET_DEPTH`` (4).
    dim:
        Model / embedding dimension. Default: ``SETUP_NET_DIM`` (512).
    heads:
        Number of attention heads. Default: ``SETUP_NET_HEADS`` (8).
    ff:
        Feed-forward hidden dimension. Default: ``SETUP_NET_FF`` (2048).
    pos_emb_std:
        Initial std for the learned absolute positional embedding.
        Default: ``SETUP_NET_POS_EMB_INIT_STD`` (0.1).
    num_piece_types:
        Number of piece types (vocabulary size). Default: ``NUM_PIECE_TYPES`` (12).
    causal:
        If ``True`` (default), apply causal masking during self-attention so
        that position ``i`` only attends to positions ``0..i``.
    """

    def __init__(
        self,
        depth: int = SETUP_NET_DEPTH,
        dim: int = SETUP_NET_DIM,
        heads: int = SETUP_NET_HEADS,
        ff: int = SETUP_NET_FF,
        pos_emb_std: float = SETUP_NET_POS_EMB_INIT_STD,
        num_piece_types: int = NUM_PIECE_TYPES,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.num_piece_types = num_piece_types
        self.token_emb = nn.Embedding(num_piece_types, dim)
        self.pos_emb = PositionalEmbedding(
            max_len=TOTAL_PIECES,
            embed_dim=dim,
            init_std=pos_emb_std,
            init="normal",
        )
        self.blocks = TransformerStack(
            depth=depth,
            embed_dim=dim,
            num_heads=heads,
            ff_dim=ff,
            dropout=0.0,
            causal=causal,
        )
        self.value_head = ValueHead(dim)
        self.entropy_head = EntropyHead(dim)
        self.policy_head = PieceDistributionHead(dim, num_types=num_piece_types)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass over a partial setup sequence.

        Parameters
        ----------
        tokens:
            Integer piece-type indices of shape ``(B, S)`` with ``S <= 40``.
            Each entry is in ``[0, num_piece_types)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            ``(value_logits, entropy, policy_logits)`` where ``value_logits``
            is ``(B, 3)`` (win/loss/draw), ``entropy`` is ``(B,)``, and
            ``policy_logits`` is ``(B, S, num_piece_types)``.
        """
        x = self.token_emb(tokens)  # (B, S, dim)
        x = self.pos_emb(x)
        x = self.blocks(x)  # causal self-attention
        value = self.value_head(x)  # (B, 3)
        entropy = self.entropy_head(x)  # (B,)
        policy = self.policy_head(x)  # (B, S, num_piece_types)
        return value, entropy, policy
