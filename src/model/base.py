from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.config import ModelConfig, Transformer


def repeat_kv_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Expand grouped K/V heads explicitly for non-CUDA SDPA backends."""
    if x.size(1) == num_heads:
        return x
    if num_heads % x.size(1) != 0:
        raise ValueError(f"Cannot expand {x.size(1)} K/V heads to {num_heads} query heads")
    return x.repeat_interleave(num_heads // x.size(1), dim=1)


def grouped_query_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Use native CUDA GQA and an explicit repeated-K/V equivalent elsewhere."""
    if q.device.type == "cuda":
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, enable_gqa=True
        )
    k = repeat_kv_heads(k, q.size(1))
    v = repeat_kv_heads(v, q.size(1))
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)


def assert_cuda_gqa_equivalence(device: torch.device) -> None:
    """Cheap startup guard that checks native GQA against repeated-K/V SDPA."""
    if device.type != "cuda":
        return
    generator = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(2, 8, 8, 16, device=device, dtype=torch.float32, generator=generator)
    k = torch.randn(2, 2, 8, 16, device=device, dtype=torch.float32, generator=generator)
    v = torch.randn(2, 2, 8, 16, device=device, dtype=torch.float32, generator=generator)
    mask = torch.ones(2, 1, 8, 8, device=device, dtype=torch.bool).tril()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.MATH):
        native = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        repeated = F.scaled_dot_product_attention(
            q, repeat_kv_heads(k, 8), repeat_kv_heads(v, 8), attn_mask=mask
        )
    torch.testing.assert_close(native, repeated, atol=1e-5, rtol=1e-5)

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 65536) -> None:
        super().__init__()

        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer("cos", theta.cos(), persistent=False)
        self.register_buffer("sin", theta.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cos = self.cos[None, : x.size(-3), None, :]
        sin = self.sin[None, : x.size(-3), None, :]
        x1, x2 = x.float().chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x)


class Attention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        num_key_value: int,
        bias: bool,
    ) -> None:
    
        super().__init__()

        assert embed_dim % num_heads == 0
        assert num_heads % num_key_value == 0

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.num_key_value = num_key_value
        self.head_dim = embed_dim // num_heads

        self.rotary = Rotary(embed_dim // num_heads)

        self.c_attn_q = nn.Linear(
            in_features=embed_dim, 
            out_features=embed_dim,
            bias=bias)

        self.c_attn_kv = nn.Linear(
            in_features=embed_dim, 
            out_features=2 * num_key_value * self.head_dim,
            bias=bias)

        self.c_proj = nn.Linear(
            in_features=embed_dim, 
            out_features=embed_dim, 
            bias=bias)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        B, T, E = x.size()

        q = self.c_attn_q(x)
        q = q.view(B, T, self.num_heads, self.head_dim)

        k = self.c_attn_kv(x)
        k, v = k.split(self.num_key_value * self.head_dim, dim=2)

        k = k.view(B, T, self.num_key_value, self.head_dim)
        v = v.view(B, T, self.num_key_value, self.head_dim)
        q, k = self.rotary(q), self.rotary(k)  # rotary expects (B, T, H, D)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        y = grouped_query_attention(q, k, v, attn_mask=attn_mask)
        y = y.transpose(1, 2).contiguous().view(B, T, E)
        y = self.c_proj(y)

        return y


class MLP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        mlp_dim: int,
    ) -> None:

        super().__init__()

        self.c_fc = nn.Linear(
            in_features=embed_dim,
            out_features=mlp_dim,
            bias=True)
            
        self.c_proj = nn.Linear(
            in_features=mlp_dim,
            out_features=embed_dim,
            bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.c_fc(x)
        h = F.gelu(h, approximate="tanh")
        h = self.c_proj(h)
        return h

class Block(nn.Module):

    def __init__(
        self, 
        embed_dim: int,
        num_heads: int,
        num_key_value: int,
        attn_bias: bool,
        mlp_dim: int,
    ) -> None:

        super().__init__()

        self.attn = Attention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_key_value=num_key_value,
            bias=attn_bias,
        )

        self.mlp = MLP(
            embed_dim=embed_dim,
            mlp_dim=mlp_dim,
        )

        self.norm_1 = nn.RMSNorm(embed_dim)
        self.norm_2 = nn.RMSNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:

        x = x + self.attn(self.norm_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm_2(x))
        return x


class BaseTransformer(Transformer):
    def __init__(
        self, 
        config: ModelConfig,
    ) -> None:

        super().__init__(config)

        self.embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.norm = nn.RMSNorm(config.embed_dim)
        self.unembed = nn.Linear(config.embed_dim, config.vocab_size, bias=True)

        self.blocks = nn.ModuleList(
            [Block(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                num_key_value=config.num_key_value,
                attn_bias=config.attn_bias,
                mlp_dim=config.mlp_dim,
            ) for _ in range(config.num_layers)])

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if hasattr(module, "bias") and getattr(module, "bias") is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.RMSNorm):
            module.weight.data.fill_(1.0)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
        stop_at_layer: int | None = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        loss = None

        def run_stack(
            tok: torch.Tensor, # (B, T)
            stop_at_layer: int | None = None,
        ) -> torch.Tensor:

            attn_mask = make_attention_mask(tok, self.config.eos_token_id)
            h = self.embed(tok) # (B, T, E)
            for i, block in enumerate(self.blocks):
                h = block(h, attn_mask=attn_mask)
                if stop_at_layer is not None and i == stop_at_layer:
                    return h
            h = self.norm(h)
            return self.unembed(h)
        
        logits = run_stack(tokens, stop_at_layer) # (B, T, V)

        if stop_at_layer is not None:
            return logits, loss

        if targets is not None:
            #calculate CE loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1000,
                reduction="mean",
            )
       
        return logits, loss


def make_attention_mask(tokens: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """Causal segmentation-aware mask. Shape (B, 1, T, T), False = masked."""
    seq_len = tokens.shape[1]
    device = tokens.device
    eos_mask = tokens == eos_token_id
    seg_ids = torch.cumsum(eos_mask.int(), dim=1) - eos_mask.int()
    same_segment = seg_ids.unsqueeze(2).eq(seg_ids.unsqueeze(1))
    causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
    allowed = same_segment & causal
    return allowed.unsqueeze(1)
