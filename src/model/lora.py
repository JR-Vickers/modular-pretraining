from typing import Generator, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import RoutedModelConfig, Transformer
from src.model.base import make_attention_mask, Attention
from src.model.utils import calc_lora_rank


def freeze(module: nn.Module, do_freeze: bool):
    """Return `module` unchanged if do_freeze=False, else a callable that runs
    the module via functional_call with all params/buffers detached.
    Activations still flow through (so downstream experts can train), but
    the module's own parameters don't accumulate .grad."""
    if not do_freeze:
        return module
    params_and_bufs = {
        **{n: p.detach() for n, p in module.named_parameters()},
        **{n: b for n, b in module.named_buffers()},
    }
    def call(*args, **kwargs):
        return torch.func.functional_call(module, params_and_bufs, args, kwargs)
    return call


class LoRA(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int = 16,
    ) -> None:

        super().__init__()
        self.A = nn.Parameter(torch.empty(in_dim, rank))
        self.B = nn.Parameter(torch.empty(rank, out_dim))
        nn.init.normal_(self.A, mean=0.0, std=0.02)
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.A @ self.B


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        rank: int = 16,
        num_experts: int = 1,
    ) -> None:

        super().__init__()
        self.experts = nn.ModuleList(
            [nn.Linear(in_dim, out_dim, bias=bias)] +  # core (idx 0)
            [LoRA(in_dim, out_dim, rank)               # aux (idx 1..E-1)
             for _ in range(num_experts - 1)]
        )

    def forward(
        self, x: torch.Tensor,
        fwd_mask: torch.Tensor,
        bck_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parallel computation over experts.

        x:        (B, T, in_dim)
        fwd_mask: (K,) per-expert forward weights. Boolean (multi-hot) during
                  training; may be float in [0,1] for capability titration, in
                  which case expert i's output (the LoRA delta for aux experts)
                  is scaled by fwd_mask[i]. Note the adapter sits on both c_fc
                  and c_proj, so a fractional weight scales each independently.
        bck_mask: (K,) boolean, experts that should receive gradient.
                  Experts not in bck_mask are wrapped via functional_call with
                  detached params — activations flow through, but their params
                  don't accumulate .grad.
        """
        y = None
        for i in range(len(self.experts)):
            w = fwd_mask[i]
            if bool(w):  # nonzero forward weight
                expert = freeze(self.experts[i], do_freeze=not bool(bck_mask[i]))
                out = expert(x)
                if not bool(w == 1):  # scale for fractional titration weights
                    out = w * out
                y = out if y is None else y + out
        return y


class MLP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        mlp_dim: int,
        num_experts: int,
        lora_rank: int,
    ) -> None:

        super().__init__()

        lora_args = {
            "num_experts": num_experts,
            "rank": lora_rank,
        }

        self.c_fc = LoRALinear(
            in_dim=embed_dim,
            out_dim=mlp_dim,
            bias=True,
            **lora_args)

        self.c_proj = LoRALinear(
            in_dim=mlp_dim,
            out_dim=embed_dim,
            bias=True,
            **lora_args)

    def forward(
        self, x: torch.Tensor,
        fwd_mask: torch.Tensor,
        bck_mask: torch.Tensor,
    ) -> torch.Tensor:

        h = self.c_fc(x, fwd_mask=fwd_mask, bck_mask=bck_mask)
        h = F.gelu(h, approximate="tanh")
        h = self.c_proj(h, fwd_mask=fwd_mask, bck_mask=bck_mask)
        return h


class Block(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_key_value: int,
        attn_bias: bool,
        num_experts: int,
        mlp_dim: int,
        lora_rank: int,
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
            num_experts=num_experts,
            lora_rank=lora_rank,
        )

        self.norm_1 = nn.RMSNorm(embed_dim)
        self.norm_2 = nn.RMSNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        fwd_mask: torch.Tensor,
        bck_mask: torch.Tensor,
    ) -> torch.Tensor:

        # core is always expert 0 (LoRALinear constructor convention)
        freeze_core = not bck_mask[0]
        attn = freeze(self.attn, do_freeze=freeze_core)
        norm_1 = freeze(self.norm_1, do_freeze=freeze_core)
        norm_2 = freeze(self.norm_2, do_freeze=freeze_core)

        x = x + attn(norm_1(x), attn_mask=attn_mask)
        x = x + self.mlp(norm_2(x), fwd_mask=fwd_mask, bck_mask=bck_mask)
        return x


class LoRATransformer(Transformer):
    def __init__(
        self,
        config: RoutedModelConfig,
        labels: list[str],
    ) -> None:

        super().__init__(config)

        embed_dim = config.embed_dim
        align = 64
        core_dim = round(config.core_param_prc * config.mlp_dim / align) * align
        aux_dim = round(config.aux_param_prc * config.mlp_dim / align) * align
        rank = calc_lora_rank(embed_dim, core_dim, aux_dim)

        self.labels = labels
        self.embed = nn.Embedding(config.vocab_size, embed_dim)
        self.norm = nn.RMSNorm(embed_dim)
        self.unembed = nn.Linear(embed_dim, config.vocab_size, bias=True)

        self.blocks = []
        for _ in range(config.num_layers):
            self.blocks.append(Block(
                embed_dim=embed_dim,
                num_heads=config.num_heads,
                num_key_value=config.num_key_value,
                attn_bias=config.attn_bias,
                num_experts=len(labels),
                mlp_dim=core_dim,
                lora_rank=rank,
            ))
        self.blocks = nn.ModuleList(self.blocks)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if hasattr(module, "bias") and getattr(module, "bias") is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.RMSNorm):
            module.weight.data.fill_(1.0)

    def get_params(self, label: str) -> Generator[torch.Tensor, None, None]:

        assert label in self.labels

        if label == "core":
            yield from self.embed.parameters()
            yield from self.norm.parameters()
            yield from self.unembed.parameters()

        for block in self.blocks:

            if label == "core":
                yield from block.attn.parameters()
                yield from block.norm_1.parameters()
                yield from block.norm_2.parameters()

            e_idx = self.labels.index(label)
            yield from block.mlp.c_fc.experts[e_idx].parameters()
            yield from block.mlp.c_proj.experts[e_idx].parameters()

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor],
        fwd_mask: torch.Tensor,
        bck_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # core is always expert 0 (LoRALinear constructor convention)
        freeze_core = not bck_mask[0]
        embed = freeze(self.embed, do_freeze=freeze_core)
        norm = freeze(self.norm, do_freeze=freeze_core)
        unembed = freeze(self.unembed, do_freeze=freeze_core)

        def run_stack(
            tok: torch.Tensor,
            fwd_mask: torch.Tensor,
            bck_mask: torch.Tensor,
        ) -> torch.Tensor:

            attn_mask = make_attention_mask(tok, self.config.eos_token_id)
            h = embed(tok)
            for block in self.blocks:
                h = block(
                    h, attn_mask=attn_mask,
                    fwd_mask=fwd_mask, bck_mask=bck_mask,
                )
            h = norm(h)
            return unembed(h)

        logits = run_stack(tokens, fwd_mask=fwd_mask, bck_mask=bck_mask) # (B, T, V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1000,
                reduction="mean",
            )

        return logits, loss
