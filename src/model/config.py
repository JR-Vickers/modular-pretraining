import dataclasses
from dataclasses import dataclass
from transformers import AutoTokenizer
from torch import nn
from typing import Literal
from src.run.util.tools import json_safe

@dataclass
class ModelConfig:

    arch: str = "base"
    tokenizer: AutoTokenizer | None = None
    ctx_len: int = 1024
    vocab_size: int = 50304
    num_layers: int = 8
    num_heads: int = 8
    num_key_value: int = 2
    attn_bias: bool = True
    eos_token_id: int = -1
    embed_dim: int = 512
    mlp_dim: int = 512 * 4
    

@dataclass
class RoutedModelConfig(ModelConfig):
    """Configuration for routed model."""
    arch: Literal["moe", "lora", "demix"] = "moe"
    core_param_prc: float = 0.95
    aux_param_prc: float = 0.05

    @classmethod
    def from_base(cls, base: ModelConfig, **overrides) -> "RoutedModelConfig":
        """Create a RoutedModelConfig inheriting all values from a ModelConfig."""
        base_fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
        base_fields.update(overrides)
        return cls(**base_fields)

class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config