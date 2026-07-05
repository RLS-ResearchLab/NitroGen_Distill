"""The NitroGen policy: raw frames -> gamepad action chunk, via flow matching.

Dataflow (one denoising step):

    frames (B, F, 3, 256, 256)
        -> SigLIP vision tower                  -> visual tokens (B, F, 256, D_vl)
        -> scatter into VL token sequence (+ optional game-ID embedding)
        -> SelfAttentionTransformer (VL mixing) -> context (B, S, D_vl)

    noisy actions (B, H, A) + timestep t
        -> MultiEmbodimentActionEncoder         -> action tokens (B, H, D)
        -> DiT cross-attending into context     -> hidden (B, H, D)
        -> action decoder MLP                   -> velocity (B, H, A)

Training minimizes MSE between predicted velocity and (actions - noise).
Inference integrates x' = v(x, t) with Euler steps from t=0 (noise) to t=1 (actions).

State-dict compatible with the official ``ng.pt``: module attribute names mirror the
official implementation exactly.
"""

from pathlib import Path
from typing import Optional

import yaml
import torch
import torch.nn.functional as F
from einops import rearrange
from pydantic import BaseModel, Field
from torch import nn
from torch.distributions import Beta

from ngd.constants import ACT_TOKEN, GAME_ID_TOKEN, IMG_SEP_TOKEN, IMG_TOKEN
from ngd.model.dit import DiT, DiTConfig, SelfAttentionTransformer, SelfAttentionTransformerConfig


class NitroGenConfig(BaseModel):
    """Field names/defaults mirror the official NitroGen_Config so the pydantic-validated
    ``ckpt_config`` stored inside ng.pt round-trips unchanged."""

    model_type: str = Field(default="nitrogen", frozen=True)

    add_pos_embed: bool = Field(default=False, description="Add learned positional embedding to action tokens.")
    model_dtype: str = Field(default="float32")
    diffusion_model_cfg: DiTConfig = Field(..., description="DiT action head config.")
    vl_self_attention_cfg: SelfAttentionTransformerConfig = Field(..., description="VL mixing transformer config.")
    hidden_size: int = Field(default=1024, description="Action-token embedding width (DiT width).")
    max_seq_len: int = Field(default=1024)
    action_dim: int = Field(default=None, description="Packed action dimension (buttons + joysticks, padded).")
    action_horizon: int = Field(default=None, description="Actions predicted per chunk.")
    noise_beta_alpha: float = Field(default=1.5)
    noise_beta_beta: float = Field(default=1.0)
    noise_s: float = Field(default=0.999, description="Scale for flow-matching time sampling.")
    num_timestep_buckets: int = Field(default=1000, description="Discretization buckets for t in [0,1].")
    num_inference_timesteps: int = Field(default=None, description="Euler steps at inference (teacher: 16).")
    max_num_embodiments: int = Field(default=1)
    vision_encoder_name: str = Field(default="google/siglip-large-patch16-256")
    vision_hidden_size: int = Field(default=768, description="Vision tower output width (D_vl).")
    add_view_embed: bool = Field(default=False)

    # Per-module trainability flags (used for fine-tuning; distillation code may override).
    tune_vision_tower: bool = Field(default=True)
    tune_mm_projector: bool = Field(default=True)
    tune_diffusion_model: bool = Field(default=True)
    tune_multi_projector: bool = Field(default=True)
    tune_vl_mixing: bool = Field(default=True)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "NitroGenConfig":
        with open(yaml_path) as f:
            return cls.model_validate(yaml.safe_load(f))


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    """(B, T) timesteps -> (B, T, embedding_dim) sin/cos features."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=timesteps.device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        freqs = timesteps.unsqueeze(-1) * exponent.exp()  # (B, T, half_dim)
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


class CategorySpecificLinear(nn.Module):
    """A bank of per-category linear layers; rows are selected by ``cat_ids``.

    NitroGen uses a single category (one embodiment: the gamepad), but the structure is
    kept for checkpoint compatibility.
    """

    def __init__(self, num_categories: int, input_dim: int, hidden_dim: int):
        super().__init__()
        self.num_categories = num_categories
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim), cat_ids: (B,)
        return torch.bmm(x, self.W[cat_ids]) + self.b[cat_ids].unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories: int, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        return self.layer2(F.relu(self.layer1(x, cat_ids)), cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    """Embed a (noisy) action chunk together with the flow timestep.

    actions (B, T, action_dim) + timestep (B,) -> action tokens (B, T, hidden_size)
    """

    def __init__(self, action_dim: int, hidden_size: int, num_embodiments: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        B, T, _ = actions.shape
        if not (timesteps.dim() == 1 and timesteps.shape[0] == B):
            raise ValueError("Expected `timesteps` of shape (B,).")
        timesteps = timesteps.unsqueeze(1).expand(-1, T)  # (B, T)

        a_emb = self.W1(actions, cat_ids)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
        x = swish(self.W2(torch.cat([a_emb, tau_emb], dim=-1), cat_ids))
        return self.W3(x, cat_ids)


class NitroGenModel(nn.Module):
    config_class = NitroGenConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: NitroGenConfig,
        game_mapping: Optional[dict] = None,  # {game_name: id}; id 0 = unconditional
        pretrained_vision: bool = True,  # False skips the SigLIP weight download (e.g. when a full ckpt is loaded right after)
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.vision_hidden_size = config.vision_hidden_size

        self.vision_encoder = self._build_vision_encoder(config.vision_encoder_name, pretrained_vision)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        # Named `model` (not `dit`) for checkpoint compatibility.
        self.model = DiT(config=config.diffusion_model_cfg)
        self.vl_self_attention_model = SelfAttentionTransformer(config=config.vl_self_attention_cfg)

        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.hidden_size,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.mm_projector = None  # unused in released checkpoints, kept for compatibility

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.hidden_size)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # Game-ID conditioning: a learned embedding placed as one token in the VL sequence.
        self.game_mapping = game_mapping
        if game_mapping is not None:
            self.game_embedding = nn.Embedding(
                len(game_mapping),
                self.vision_hidden_size,
                padding_idx=0,  # 0 = unconditional
                scale_grad_by_freq=True,
            )

        self.set_trainable_parameters(
            tune_multi_projector=config.tune_multi_projector,
            tune_diffusion_model=config.tune_diffusion_model,
            tune_vision_tower=config.tune_vision_tower,
            tune_mm_projector=config.tune_mm_projector,
            tune_vl_mixing=config.tune_vl_mixing,
        )

    @staticmethod
    def _build_vision_encoder(name: str, pretrained: bool) -> nn.Module:
        if "siglip" not in name:
            from transformers import AutoModel

            return AutoModel.from_pretrained(name)

        from transformers import SiglipVisionConfig, SiglipVisionModel

        if pretrained:
            model = SiglipVisionModel.from_pretrained(name)
        else:
            model = SiglipVisionModel(SiglipVisionConfig.from_pretrained(name))
        return model.vision_model

    # ------------------------------------------------------------------ freezing

    def set_trainable_parameters(
        self,
        tune_multi_projector: bool = True,
        tune_diffusion_model: bool = True,
        tune_vision_tower: bool = True,
        tune_mm_projector: bool = True,
        tune_vl_mixing: bool = True,
    ):
        self.tune_multi_projector = tune_multi_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vision_tower = tune_vision_tower
        self.tune_mm_projector = tune_mm_projector
        self.tune_vl_mixing = tune_vl_mixing

        for param in self.parameters():
            param.requires_grad = True

        # The official implementation freezes the last SigLIP block (index 11) and the
        # pooling head: their outputs are unused since features are taken pre-pooling.
        if hasattr(self.vision_encoder, "encoder"):
            for param in self.vision_encoder.encoder.layers[11].parameters():
                param.requires_grad = False
        if hasattr(self.vision_encoder, "head"):
            for param in self.vision_encoder.head.parameters():
                param.requires_grad = False

        if not tune_multi_projector:
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        if not tune_vision_tower:
            self.vision_encoder.requires_grad_(False)
        if self.mm_projector is not None and not tune_mm_projector:
            self.mm_projector.requires_grad_(False)
        if not tune_vl_mixing:
            self.vl_self_attention_model.requires_grad_(False)

    def set_frozen_modules_to_eval_mode(self):
        """Keep frozen modules in eval mode during training (dropout/batchnorm correctness)."""
        if not self.training:
            return
        if not self.tune_multi_projector:
            self.action_encoder.eval()
            self.action_decoder.eval()
        if not self.tune_diffusion_model:
            self.model.eval()
        if not self.tune_vision_tower:
            self.vision_encoder.eval()
        if self.mm_projector is not None and not self.tune_mm_projector:
            self.mm_projector.eval()
        if not self.tune_vl_mixing:
            self.vl_self_attention_model.eval()

    # ------------------------------------------------------------------ encoding

    def sample_time(self, batch_size: int, device, dtype) -> torch.Tensor:
        """Sample flow-matching time t in [0, noise_s], biased toward t=0 (high noise)."""
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (1 - sample) * self.config.noise_s

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """(B, F, C, H, W) frames -> (B, F, tokens_per_frame, D_vl) visual tokens."""
        batch_size, num_frames, channels, height, width = images.shape
        images = images.reshape(-1, channels, height, width)
        image_features = self.vision_encoder(images)["last_hidden_state"]
        image_features = rearrange(image_features, "(b f) n d -> b f n d", f=num_frames)
        if self.mm_projector is not None:
            image_features = self.mm_projector(image_features)
        return image_features

    def prepare_input_embs(
        self,
        vl_token_ids: torch.Tensor,  # (B, S) token-type ids for the VL sequence
        sa_token_ids: torch.Tensor,  # (B, T) token-type ids for the state-action sequence
        vision: torch.Tensor,  # (B, F, tokens_per_frame, D_vl)
        action: torch.Tensor,  # (B, T, D) embedded (noisy) actions
        dropped_images: torch.Tensor,  # (B, F) 1 where the frame slot is padding
        game_ids: Optional[torch.Tensor] = None,  # (B,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scatter visual/game/action embeddings into their token slots.

        Returns (vl_embs (B, S, D_vl), sa_embs (B, T, D)).
        """
        B, S = vl_token_ids.shape
        vl_embs = torch.zeros(B, S, self.vision_hidden_size, dtype=vision.dtype, device=vision.device)

        # --- visual tokens: place non-dropped frames at IMG_TOKEN positions
        B, num_images, tokens_per_image, _ = vision.shape
        vision_mask = vl_token_ids == IMG_TOKEN  # (B, S)
        vision_flat = vision.reshape(B, -1, self.vision_hidden_size)
        non_dropped = (dropped_images == 0).unsqueeze(-1).repeat(1, 1, tokens_per_image).reshape(B, -1)
        valid_vision_embs = vision_flat[non_dropped]

        assert valid_vision_embs.shape[0] == vision_mask.sum().item(), (
            f"{valid_vision_embs.shape[0]} valid visual embeddings but "
            f"{vision_mask.sum().item()} IMG_TOKEN slots"
        )
        batch_idx, token_idx = vision_mask.nonzero(as_tuple=True)
        vl_embs[batch_idx, token_idx] = valid_vision_embs

        # --- game-ID token
        if self.game_mapping is not None and game_ids is not None:
            game_mask = vl_token_ids == GAME_ID_TOKEN
            if game_mask.any():
                per_batch = game_mask.sum(dim=1)
                assert torch.all(per_batch == 1), f"Expected exactly one game token per item, got {per_batch.tolist()}"
                game_embs = self.game_embedding(game_ids)  # (B, D_vl)
                batch_idx, token_idx = game_mask.nonzero(as_tuple=True)
                vl_embs[batch_idx, token_idx] = game_embs[batch_idx].to(dtype=vl_embs.dtype)

        if (vl_token_ids == IMG_SEP_TOKEN).any():
            raise NotImplementedError("IMG_SEP tokens are not used by released checkpoints.")

        # --- action tokens
        B, T = sa_token_ids.shape
        sa_embs = torch.zeros(B, T, self.hidden_size, dtype=vision.dtype, device=vision.device)
        action_mask = (sa_token_ids == ACT_TOKEN).unsqueeze(-1).expand_as(sa_embs)
        sa_embs = sa_embs.masked_scatter(action_mask, action)

        if self.config.add_pos_embed:
            pos_ids = torch.arange(T, dtype=torch.long, device=sa_token_ids.device)
            sa_embs = sa_embs + self.position_embedding(pos_ids).unsqueeze(0)

        return vl_embs, sa_embs

    def _predict_velocity(
        self,
        actions: torch.Tensor,  # (B, H, A) current noisy actions
        t_discretized: int,
        visual_features: torch.Tensor,
        data: dict,
        embodiment_id: torch.Tensor,
    ) -> torch.Tensor:
        """One DiT evaluation: embed actions at time t, mix VL context, predict velocity."""
        device = actions.device
        timesteps = torch.full((actions.shape[0],), t_discretized, dtype=torch.long, device=device)

        action_features = self.action_encoder(actions, timesteps, embodiment_id)
        vl_embs, sa_embs = self.prepare_input_embs(
            data["vl_token_ids"],
            data["sa_token_ids"],
            visual_features,
            action_features,
            data["dropped_images"],
            game_ids=data.get("game_ids"),
        )
        vl_embs = self.vl_self_attention_model(vl_embs)
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=timesteps,
        )
        pred = self.action_decoder(model_output, embodiment_id)
        return pred[:, -actions.shape[1]:]

    # ------------------------------------------------------------------ training

    def forward(self, data: dict) -> dict:
        """Flow-matching training loss on one batch (see module docstring for keys)."""
        self.set_frozen_modules_to_eval_mode()

        embodiment_id = data["embodiment_id"]
        has_real_action = data["has_real_action"]

        visual_features = self.encode_images(data["images"])

        # Noisy interpolant x_t = (1-t) * noise + t * actions; target velocity = actions - noise.
        actions = data["actions"]
        noise = torch.randn_like(actions)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]
        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()

        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)
        vl_embs, sa_embs = self.prepare_input_embs(
            data["vl_token_ids"],
            data["sa_token_ids"],
            visual_features,
            action_features,
            data["dropped_images"],
            game_ids=data.get("game_ids", data.get("game_id")),
        )
        vl_embs = self.vl_self_attention_model(vl_embs)
        model_output, all_hidden_states = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=t_discretized,
            return_all_hidden_states=True,
        )
        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1]:]

        # Masked velocity MSE (mask covers padded action dims and fake-action samples).
        mask = has_real_action[:, None, None] * data["actions_mask"]
        raw_loss = F.mse_loss(pred_actions, velocity, reduction="none") * mask
        loss = (has_real_action[:, None, None] * raw_loss).sum() / (mask.sum() + 1e-6)

        return {"loss": loss}

    # ------------------------------------------------------------------ inference

    @torch.inference_mode()
    def get_action(self, data: dict, num_inference_timesteps: Optional[int] = None) -> dict:
        """Sample an action chunk with Euler integration of the learned velocity field.

        ``num_inference_timesteps`` overrides the checkpoint default (16) -- useful for
        step-count ablations and distillation baselines.
        """
        embodiment_id = data["embodiment_id"]
        batch_size = data["images"].shape[0]
        device, dtype = data["images"].device, data["images"].dtype

        actions = torch.randn(batch_size, self.config.action_horizon, self.config.action_dim, dtype=dtype, device=device)

        num_steps = num_inference_timesteps or self.num_inference_timesteps
        dt = 1.0 / num_steps

        # Vision context does not depend on t: encode once.
        visual_features = self.encode_images(data["images"])

        for i in range(num_steps):
            t_discretized = int(i / num_steps * self.num_timestep_buckets)
            velocity = self._predict_velocity(actions, t_discretized, visual_features, data, embodiment_id)
            actions = actions + dt * velocity

        return {"action_tensor": actions}

    @torch.inference_mode()
    def get_action_with_cfg(
        self,
        data_cond: dict,
        data_uncond: dict,
        cfg_scale: float = 1.0,
        num_inference_timesteps: Optional[int] = None,
    ) -> dict:
        """Classifier-free-guided sampling: extrapolate from the unconditional velocity
        toward the conditional one (conditioning = frame history and/or game ID)."""
        embodiment_id = data_cond["embodiment_id"]
        batch_size = data_cond["images"].shape[0]
        device, dtype = data_cond["images"].device, data_cond["images"].dtype

        actions = torch.randn(batch_size, self.config.action_horizon, self.config.action_dim, dtype=dtype, device=device)

        num_steps = num_inference_timesteps or self.num_inference_timesteps
        dt = 1.0 / num_steps

        visual_cond = self.encode_images(data_cond["images"])
        visual_uncond = self.encode_images(data_uncond["images"])

        for i in range(num_steps):
            t_discretized = int(i / num_steps * self.num_timestep_buckets)
            v_cond = self._predict_velocity(actions, t_discretized, visual_cond, data_cond, embodiment_id)
            v_uncond = self._predict_velocity(actions, t_discretized, visual_uncond, data_uncond, embodiment_id)
            velocity = v_cond + cfg_scale * (v_cond - v_uncond)
            actions = actions + dt * velocity

        return {"action_tensor": actions}

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
