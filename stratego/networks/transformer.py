"""Pre-LN transformer building blocks for the Stratego networks.

Implements the architectural primitives shared by the setup, move, and belief
networks (arXiv:2511.07312).  The paper specifies pre-layernorm transformer
blocks ("layernorm before the sublayer rather than after the residual") with
learned absolute positional embeddings.

This module provides only the building blocks; the full setup/move/belief
networks are assembled in later tasks.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from stratego.networks.attention import MultiHeadAttention

__all__ = [
    "EntropyHead",
    "PieceDistributionHead",
    "PositionalEmbedding",
    "PreLNTransformerBlock",
    "TransformerStack",
    "ValueHead",
]


class PositionalEmbedding(nn.Module):
    """Learned absolute positional embeddings.

    Paper: "learned absolute positional embeddings" with init std=0.1 (setup/move
    nets) or Kaiming uniform (belief net).

    Parameters
    ----------
    max_len:
        Maximum sequence length the embedding can address.
    embed_dim:
        Embedding / model dimension.
    init_std:
        Standard deviation for the normal initializer (used when ``init="normal"``).
    init:
        Initializer — ``"normal"`` (default, std=``init_std``) or
        ``"kaiming_uniform"`` (matches the PyTorch ``nn.Linear`` default,
        ``a=sqrt(5)``).
    """

    def __init__(
        self,
        max_len: int,
        embed_dim: int,
        init_std: float = 0.1,
        init: str = "normal",
    ) -> None:
        super().__init__()
        self.embed = nn.Parameter(torch.zeros(max_len, embed_dim))
        if init == "normal":
            nn.init.normal_(self.embed, std=init_std)
        elif init == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.embed, a=5**0.5)  # matches PyTorch Linear default
        else:
            raise ValueError(f"Unknown init '{init}' (expected 'normal' or 'kaiming_uniform')")

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Add the positional embedding to ``x``.

        Parameters
        ----------
        x:
            Input of shape ``(B, S, E)``.
        offset:
            Starting position index — useful when the same embedding table is
            reused across segments (e.g. belief decoder offset by encoder length).

        Returns
        -------
        torch.Tensor:
            Tensor of shape ``(B, S, E)`` — input plus positional embedding.
        """
        return x + self.embed[offset : offset + x.size(1)].unsqueeze(0)


class PreLNTransformerBlock(nn.Module):
    """Pre-LN transformer block: ``LN -> Attn -> Residual -> LN -> FFN -> Residual``.

    Paper: "Pre-layernorm (i.e., layernorm before the sublayer rather than after
    the residual)".

    Parameters
    ----------
    embed_dim:
        Model / embedding dimension.
    num_heads:
        Number of attention heads.
    ff_dim:
        Feed-forward hidden dimension.
    dropout:
        Dropout probability applied to attention and FFN outputs.  ``0.0`` uses
        an :class:`~torch.nn.Identity` (no overhead).
    causal:
        If ``True``, apply causal masking during self-attention.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.0,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout=dropout, causal=causal)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.dropout: nn.Module = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, key_value: torch.Tensor | None = None) -> torch.Tensor:
        """Pre-LN forward pass.

        Parameters
        ----------
        x:
            Input of shape ``(B, S, E)``.
        key_value:
            Optional cross-attention source of shape ``(B, S_kv, E)``.  When
            ``None``, self-attention is performed.

        Returns
        -------
        torch.Tensor:
            Output of shape ``(B, S, E)``.
        """
        # Pre-LN: normalize BEFORE the sublayer, residual AFTER.
        x = x + self.dropout(self.attn(self.ln1(x), key_value))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x


class TransformerStack(nn.Module):
    """Stack of pre-LN transformer blocks.

    Parameters
    ----------
    depth:
        Number of :class:`PreLNTransformerBlock` layers.
    embed_dim, num_heads, ff_dim, dropout, causal:
        Forwarded to every block.
    """

    def __init__(
        self,
        depth: int,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.0,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [PreLNTransformerBlock(embed_dim, num_heads, ff_dim, dropout, causal) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor, key_value: torch.Tensor | None = None) -> torch.Tensor:
        """Apply each block in sequence.

        Parameters
        ----------
        x:
            Input of shape ``(B, S, E)``.
        key_value:
            Optional cross-attention source forwarded to every block.

        Returns
        -------
        torch.Tensor:
            Output of shape ``(B, S, E)``.
        """
        for block in self.blocks:
            x = block(x, key_value)
        return x


class ValueHead(nn.Module):
    """Categorical value prediction head: outputs ``(B, 3)`` for ``[win, loss, draw]``.

    When the input is 3-D ``(B, S, E)``, the first token (value token) is used.
    When the input is 2-D ``(B, E)``, it is projected directly.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``(B, 3)`` logits.

        Parameters
        ----------
        x:
            Input of shape ``(B, S, E)`` or ``(B, E)``.

        Returns
        -------
        torch.Tensor:
            Logits of shape ``(B, 3)``.
        """
        if x.dim() == 3:
            x = x[:, 0]  # take first token (value token)
        return cast(torch.Tensor, self.proj(self.ln(x)))


class EntropyHead(nn.Module):
    """Conditional entropy prediction head: outputs ``(B,)`` scalar.

    When the input is 3-D ``(B, S, E)``, the first token is used.
    When the input is 2-D ``(B, E)``, it is projected directly.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``(B,)`` entropy estimates.

        Parameters
        ----------
        x:
            Input of shape ``(B, S, E)`` or ``(B, E)``.

        Returns
        -------
        torch.Tensor:
            Tensor of shape ``(B,)``.
        """
        if x.dim() == 3:
            x = x[:, 0]
        return cast(torch.Tensor, self.proj(self.ln(x)).squeeze(-1))


class PieceDistributionHead(nn.Module):
    """Next-piece distribution head for the setup network.

    Outputs ``(B, S, num_types)`` logits — a per-position distribution over
    piece types (used autoregressively during setup generation).
    """

    def __init__(self, embed_dim: int, num_types: int = 12) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, num_types)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-position logits.

        Parameters
        ----------
        x:
            Input of shape ``(B, S, E)``.

        Returns
        -------
        torch.Tensor:
            Logits of shape ``(B, S, num_types)``.
        """
        return cast(torch.Tensor, self.proj(self.ln(x)))
