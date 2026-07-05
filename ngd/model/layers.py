"""Building blocks shared by the DiT action head and the VL mixing transformer.

Attribute names (``norm1``, ``attn1``, ``norm3``, ``ff``, ``pos_embed``, ...) are kept
identical to the official implementation so that ``ng.pt`` state dicts load without
any key remapping. Do not rename module attributes here.
"""

from typing import Optional

import torch
from torch import nn
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import (
    SinusoidalPositionalEmbedding,
    TimestepEmbedding,
    Timesteps,
)


class TimestepEncoder(nn.Module):
    """Sinusoidal projection of a discrete timestep followed by a small MLP.

    (B,) int timesteps -> (B, embedding_dim) conditioning vector for AdaLN.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        return self.timestep_embedder(self.time_proj(timesteps).to(dtype))


class AdaLayerNorm(nn.Module):
    """LayerNorm whose scale/shift are regressed from the timestep embedding."""

    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, norm_eps, norm_elementwise_affine)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(self.silu(temb)).chunk(2, dim=1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: (Ada)LN -> attention -> LN -> feed-forward.

    The single attention module doubles as self- or cross-attention: when
    ``encoder_hidden_states`` is passed at forward time, ``attn1`` attends over it
    (keys/values from the encoder sequence); otherwise it self-attends.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # "layer_norm" | "ada_norm"
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        positional_embeddings: Optional[str] = None,  # None | "sinusoidal"
        num_positional_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.norm_type = norm_type

        if positional_embeddings == "sinusoidal":
            if num_positional_embeddings is None:
                raise ValueError("`num_positional_embeddings` required with sinusoidal embeddings.")
            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
        elif positional_embeddings is None:
            self.pos_embed = None
        else:
            raise ValueError(f"Unsupported positional_embeddings: {positional_embeddings}")

        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim, norm_elementwise_affine=False, norm_eps=norm_eps)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            out_bias=True,
        )

        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)
        self.final_dropout = nn.Dropout(dropout) if final_dropout else None

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, T, D)
        encoder_hidden_states: Optional[torch.Tensor] = None,  # (B, S, D_enc) for cross-attention
        temb: Optional[torch.Tensor] = None,  # (B, D) timestep embedding, required for ada_norm
    ) -> torch.Tensor:
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
        if self.final_dropout is not None:
            attn_output = self.final_dropout(attn_output)
        hidden_states = attn_output + hidden_states

        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
        return hidden_states
