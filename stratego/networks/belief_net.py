"""Belief network: encoder-decoder transformer for opponent hidden piece prediction.

Implements the belief network from arXiv:2511.07312.  The encoder processes the
full board state (infostate); the decoder autoregressively predicts the types of
opponent hidden pieces in row-major order, using cross-attention to the
encoder's hidden-piece tokens.

Architecture (from tab:belief-network-hyper):
    - Encoder: 6 pre-LN blocks, dim=512, heads=8, ff=2048, dropout=0.2
    - Decoder: 4 pre-LN blocks (nn.TransformerDecoder), dim=512, heads=8, ff=2048, dropout=0.2
    - Positional embeddings: Kaiming uniform init (unlike setup/move which use normal std=0.1)
    - Dropout 0.2 during training to generalize to OOD opponents (e.g. humans)
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from stratego.constants import (
    BELIEF_DROPOUT,
    BELIEF_NET_DECODER_BLOCKS,
    BELIEF_NET_DIM,
    BELIEF_NET_ENCODER_DEPTH,
    BELIEF_NET_FF,
    BELIEF_NET_HEADS,
    NUM_INFOSTATE_CHANNELS,
    NUM_PIECE_TYPES,
    NUM_SQUARES,
    TOTAL_PIECES,
)
from stratego.networks.transformer import PositionalEmbedding, TransformerStack

__all__ = ["BeliefNetwork"]


class BeliefNetwork(nn.Module):
    """Encoder-decoder transformer for predicting opponent hidden piece types.

    Encoder processes the full board state.  Decoder autoregressively predicts
    the types of opponent hidden pieces in row-major order, using cross-attention
    to the encoder's hidden-piece tokens.

    Parameters
    ----------
    enc_depth:
        Number of encoder transformer blocks (paper: 6).
    dec_blocks:
        Number of decoder transformer blocks (paper: 4).
    dim:
        Model / embedding dimension (paper: 512).
    heads:
        Number of attention heads (paper: 8).
    ff:
        Feed-forward hidden dimension (paper: 2048).
    dropout:
        Dropout probability applied during training (paper: 0.2).
    num_piece_types:
        Number of piece types for the output vocabulary (12).
    num_squares:
        Number of board squares for the encoder positional embedding (100).
    infostate_channels:
        Number of infostate channels -- input feature depth (488).
    """

    def __init__(
        self,
        enc_depth: int = BELIEF_NET_ENCODER_DEPTH,
        dec_blocks: int = BELIEF_NET_DECODER_BLOCKS,
        dim: int = BELIEF_NET_DIM,
        heads: int = BELIEF_NET_HEADS,
        ff: int = BELIEF_NET_FF,
        dropout: float = BELIEF_DROPOUT,
        num_piece_types: int = NUM_PIECE_TYPES,
        num_squares: int = NUM_SQUARES,
        infostate_channels: int = NUM_INFOSTATE_CHANNELS,
    ) -> None:
        super().__init__()

        # --- Encoder ------------------------------------------------------
        self.input_proj = nn.Linear(infostate_channels, dim)
        self.enc_pos_emb = PositionalEmbedding(
            max_len=num_squares,
            embed_dim=dim,
            init="kaiming_uniform",
        )
        self.encoder = TransformerStack(
            depth=enc_depth,
            embed_dim=dim,
            num_heads=heads,
            ff_dim=ff,
            dropout=dropout,
            causal=False,
        )

        # --- Decoder ------------------------------------------------------
        self.token_emb = nn.Embedding(num_piece_types, dim)
        self.dec_pos_emb = PositionalEmbedding(
            max_len=TOTAL_PIECES,
            embed_dim=dim,
            init="kaiming_uniform",
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=ff,
            dropout=dropout,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_blocks)

        # --- Output projection --------------------------------------------
        self.output_proj = nn.Linear(dim, num_piece_types)

    def forward(
        self,
        infostate: torch.Tensor,
        hidden_mask: torch.Tensor,
        target_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        infostate:
            Board state of shape ``(B, 488, 10, 10)``.
        hidden_mask:
            Boolean mask of shape ``(B, 100)`` -- ``True`` for squares with
            hidden opponent pieces.  Used as the cross-attention key padding
            mask (non-hidden squares are masked out).
        target_tokens:
            Target token indices of shape ``(B, S)`` for teacher forcing.
            When ``None``, returns the encoder output ``(B, 100, dim)`` for
            external autoregressive processing.

        Returns
        -------
        torch.Tensor:
            Logits of shape ``(B, S, num_piece_types)`` when ``target_tokens``
            is provided, else encoder output ``(B, 100, dim)``.
        """
        # --- Encode the full board ----------------------------------------
        # (B, C, H, W) -> (B, H*W, C) = (B, 100, 488)
        x = infostate.permute(0, 2, 3, 1).flatten(1, 2)
        x = self.input_proj(x)  # (B, 100, dim)
        x = self.enc_pos_emb(x)
        enc_out = self.encoder(x)  # (B, 100, dim)

        if target_tokens is None:
            return cast(torch.Tensor, enc_out)

        # --- Decode with cross-attention to encoder output ----------------
        # memory_key_padding_mask: True for positions to MASK OUT (non-hidden).
        # nn.TransformerDecoder convention: True in key_padding_mask = ignore.
        memory_key_padding_mask = ~hidden_mask  # (B, 100)

        tgt = self.token_emb(target_tokens)  # (B, S, dim)
        tgt = self.dec_pos_emb(tgt)  # (B, S, dim)

        # Causal mask for autoregressive decoding: positions above the diagonal
        # are masked with -inf so token i can only attend to tokens 0..i.
        seq_len = target_tokens.shape[1]
        tgt_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=tgt.device, dtype=tgt.dtype),
            diagonal=1,
        )

        dec_out = self.decoder(
            tgt,
            enc_out,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # (B, S, dim)

        logits = self.output_proj(dec_out)  # (B, S, num_piece_types)
        return cast(torch.Tensor, logits)
