import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).parent.parent
CKPT_PATH = REPO_ROOT / "checkpoints" / "ng.pt"
OFFICIAL_REPO = REPO_ROOT / "third_party" / "NitroGen"

needs_ckpt = pytest.mark.skipif(not CKPT_PATH.exists(), reason="checkpoints/ng.pt not downloaded (./env.sh weights)")
needs_official = pytest.mark.skipif(not OFFICIAL_REPO.exists(), reason="third_party/NitroGen not cloned (./env.sh test does this)")


def official_repo_on_path():
    if str(OFFICIAL_REPO) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_REPO))


@pytest.fixture(scope="session")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def tiny_model_and_tokenizer():
    """Small random NitroGen exercising the exact same code paths as ng.pt."""
    from ngd.data.tokenizer import NitroGenTokenizer, TokenizerConfig
    from ngd.model.dit import DiTConfig, SelfAttentionTransformerConfig
    from ngd.model.nitrogen import NitroGenConfig, NitroGenModel

    config = NitroGenConfig(
        diffusion_model_cfg=DiTConfig(
            num_attention_heads=4,
            attention_head_dim=32,
            num_layers=2,
            interleave_self_attention=True,
            positional_embeddings=None,
            cross_attention_dim=768,
            output_dim=128,
        ),
        vl_self_attention_cfg=SelfAttentionTransformerConfig(
            num_attention_heads=8,
            attention_head_dim=96,
            num_layers=1,
            positional_embeddings=None,
        ),
        hidden_size=128,
        action_dim=25,
        action_horizon=18,
        num_inference_timesteps=4,
        add_pos_embed=True,
        vision_encoder_name="google/siglip2-base-patch16-256",
        vision_hidden_size=768,
    )
    torch.manual_seed(0)
    model = NitroGenModel(config, pretrained_vision=False).eval()
    tokenizer = NitroGenTokenizer(TokenizerConfig(max_sequence_length=256, action_horizon=18, training=True))
    return model, tokenizer


@pytest.fixture(scope="session")
def loaded_ckpt(device):
    from ngd.model.loading import load_checkpoint

    return load_checkpoint(CKPT_PATH, device=device, verbose=False)
