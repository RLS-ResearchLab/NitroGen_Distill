"""The two transformers of NitroGen's action head.

- ``SelfAttentionTransformer``: mixes the vision(-language) token sequence before it is
  consumed by the DiT ("VL mixing" in the paper).
- ``DiT``: flow-matching transformer over noisy action tokens, cross-attending into the
  mixed VL tokens, conditioned on the flow timestep via AdaLN. This is what runs once per
  denoising step at inference -- i.e. the main distillation target.

Attribute names must stay in sync with the official checkpoint (see layers.py docstring).
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel, Field

from ngd.model.layers import TimestepEncoder, TransformerBlock


class DiTConfig(BaseModel):
    num_attention_heads: int = Field(default=8)
    attention_head_dim: int = Field(default=64)
    output_dim: int = Field(default=26)
    num_layers: int = Field(default=12)
    dropout: float = Field(default=0.1)
    attention_bias: bool = Field(default=True)
    activation_fn: str = Field(default="gelu-approximate")
    num_embeds_ada_norm: Optional[int] = Field(default=1000)
    upcast_attention: bool = Field(default=False)
    norm_type: str = Field(default="ada_norm")
    norm_elementwise_affine: bool = Field(default=False)
    norm_eps: float = Field(default=1e-5)
    max_num_positional_embeddings: int = Field(default=512)
    compute_dtype: str = Field(default="float32")
    final_dropout: bool = Field(default=True)
    positional_embeddings: Optional[str] = Field(default="sinusoidal")
    interleave_self_attention: bool = Field(default=False, description="If True, odd layers self-attend instead of cross-attending.")
    cross_attention_dim: Optional[int] = Field(default=None, description="Width of the VL tokens for cross-attention. None disables cross-attention.")


class DiT(nn.Module):
    """Flow-matching action transformer.

    forward: noisy action embeddings (B, T, D) + VL context (B, S, D_vl) + timestep (B,)
             -> velocity prediction in model space (B, T, output_dim).
    """

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config
        self.inner_dim = config.num_attention_heads * config.attention_head_dim

        self.timestep_encoder = TimestepEncoder(embedding_dim=self.inner_dim)

        blocks = []
        for idx in range(config.num_layers):
            # With interleaving enabled, odd layers are pure self-attention layers.
            self_attn_only = config.interleave_self_attention and idx % 2 == 1
            blocks.append(
                TransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=config.num_attention_heads,
                    attention_head_dim=config.attention_head_dim,
                    dropout=config.dropout,
                    cross_attention_dim=None if self_attn_only else config.cross_attention_dim,
                    activation_fn=config.activation_fn,
                    attention_bias=config.attention_bias,
                    upcast_attention=config.upcast_attention,
                    norm_type=config.norm_type,
                    norm_elementwise_affine=config.norm_elementwise_affine,
                    norm_eps=config.norm_eps,
                    final_dropout=config.final_dropout,
                    positional_embeddings=config.positional_embeddings,
                    num_positional_embeddings=config.max_num_positional_embeddings,
                )
            )
        self.transformer_blocks = nn.ModuleList(blocks)

        # AdaLN-style output head: timestep-conditioned scale/shift, then projection.
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(self.inner_dim, config.output_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, T, D) action token embeddings
        encoder_hidden_states: torch.Tensor,  # (B, S, D_vl) mixed VL tokens
        timestep: torch.LongTensor,  # (B,) discretized flow timestep
        return_all_hidden_states: bool = False,
    ):
        # NOTE: no encoder attention mask. The official model builds a vl_attn_mask but
        # never feeds it to attention (all VL tokens, including left-padding, are attended
        # to). We match that exactly for checkpoint parity -- do not wire the mask in here.
        temb = self.timestep_encoder(timestep)

        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        all_hidden_states = [hidden_states]
        for idx, block in enumerate(self.transformer_blocks):
            self_attn_only = self.config.interleave_self_attention and idx % 2 == 1
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=None if self_attn_only else encoder_hidden_states,
                temb=temb,
            )
            all_hidden_states.append(hidden_states)

        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        out = self.proj_out_2(hidden_states)

        if return_all_hidden_states:
            return out, all_hidden_states
        return out


class SelfAttentionTransformerConfig(BaseModel):
    num_attention_heads: int = Field(default=8)
    attention_head_dim: int = Field(default=64)
    output_dim: int = Field(default=26)
    num_layers: int = Field(default=12)
    dropout: float = Field(default=0.1)
    attention_bias: bool = Field(default=True)
    activation_fn: str = Field(default="gelu-approximate")
    num_embeds_ada_norm: Optional[int] = Field(default=1000)
    upcast_attention: bool = Field(default=False)
    max_num_positional_embeddings: int = Field(default=512)
    compute_dtype: str = Field(default="float32")
    final_dropout: bool = Field(default=True)
    positional_embeddings: Optional[str] = Field(default="sinusoidal")
    interleave_self_attention: bool = Field(default=False)


class SelfAttentionTransformer(nn.Module):
    """Plain pre-norm self-attention stack used to mix VL tokens before the DiT."""

    def __init__(self, config: SelfAttentionTransformerConfig):
        super().__init__()
        self.config = config
        self.inner_dim = config.num_attention_heads * config.attention_head_dim

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=config.num_attention_heads,
                    attention_head_dim=config.attention_head_dim,
                    dropout=config.dropout,
                    activation_fn=config.activation_fn,
                    attention_bias=config.attention_bias,
                    upcast_attention=config.upcast_attention,
                    positional_embeddings=config.positional_embeddings,
                    num_positional_embeddings=config.max_num_positional_embeddings,
                    final_dropout=config.final_dropout,
                )
                for _ in range(config.num_layers)
            ]
        )

    def forward(self, hidden_states: torch.Tensor, return_all_hidden_states: bool = False):
        hidden_states = hidden_states.contiguous()
        all_hidden_states = [hidden_states]
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
            all_hidden_states.append(hidden_states)

        if return_all_hidden_states:
            return hidden_states, all_hidden_states
        return hidden_states
