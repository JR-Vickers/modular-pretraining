from typing import Generator, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.config import RoutedModelConfig, Transformer
from src.model.base import make_attention_mask, Attention, MLP


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


class MoE(nn.Module):
    """
    Gateless MoE. Dispatch by per-sample multi-hot mask over experts.
    """
    def __init__(
        self,
        embed_dim: int,
        num_experts: int,
        core_dim: int,
        aux_dim: int,
    ) -> None:

        super().__init__()

        self.experts = nn.ModuleList(
            [MLP(embed_dim, core_dim)] +        # core (idx 0)
            [MLP(embed_dim, aux_dim)           # aux  (idx 1..E-1)
             for _ in range(num_experts - 1)]
        )

    def forward(
        self, x: torch.Tensor, 
        fwd_mask: torch.Tensor,
        bck_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Runs all the experts in parallel. More efficient approach for single GPU with compiled model.
        
        x:           (B, T, E) for batch, sequence length, and embedding dimension
        fwd_mask: (K,) per-expert forward weights. Boolean (multi-hot) during
                  training; may be float in [0,1] for capability titration, in
                  which case expert i's output is scaled by fwd_mask[i].
        bck_mask: (K,) boolean, multi-hot selection of experts for the backward pass.
        returns:     (B, T, E)
        """

        # Fast path only for the pure all-active training case (all weights == 1).
        # Fractional (titration) masks fall through to the weighted slow path below.
        if bool((fwd_mask == 1).all()) and bool((bck_mask == 1).all()):

            K = len(self.experts)

            #get core output
            y = self.experts[0](x)

            # Get stacked aux weights
            aux_fc_w = torch.stack([e.c_fc.weight for e in self.experts[1:]], dim=0)
            aux_fc_b = torch.stack([e.c_fc.bias for e in self.experts[1:]], dim=0)
            aux_proj_w = torch.stack([e.c_proj.weight for e in self.experts[1:]], dim=0)
            aux_proj_b = torch.stack([e.c_proj.bias for e in self.experts[1:]], dim=0)
            
            # Batched forward: x @ W^T for all aux experts at once
            # einsum: (B,T,E) with (K-1,H,E) -> (B,T,K-1,H)
            aux_hidden = torch.einsum('bte,khe->btkh', x, aux_fc_w) + aux_fc_b.view(1, 1, K-1, -1)
            aux_hidden = F.gelu(aux_hidden, approximate="tanh")
            
            # Second layer: (B,T,K-1,H) with (K-1,E,H) -> (B,T,E)
            aux_output = torch.einsum('btkh,keh->bte', aux_hidden, aux_proj_w) + aux_proj_b.sum(dim=0)
            
            # Sum MoE outputs with core output
            return y + aux_output  # (B, T, E)

        else:

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


class Block(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_key_value: int,
        attn_bias: bool,
        num_experts: int,
        core_dim: int,
        aux_dim: int,
    ) -> None:

        super().__init__()

        self.attn = Attention(
            num_heads=num_heads,
            embed_dim=embed_dim,
            num_key_value=num_key_value,
            bias=attn_bias,
        )
    
        self.moe = MoE(
            embed_dim=embed_dim,
            core_dim=core_dim,
            aux_dim=aux_dim,
            num_experts=num_experts,
        )

        self.norm_1 = nn.RMSNorm(embed_dim)
        self.norm_2 = nn.RMSNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        fwd_mask: torch.Tensor, # (K,)
        bck_mask: torch.Tensor, # (K,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        freeze_core = not bck_mask[0]
        attn = freeze(self.attn, do_freeze=freeze_core)
        norm_1 = freeze(self.norm_1, do_freeze=freeze_core)
        norm_2 = freeze(self.norm_2, do_freeze=freeze_core)

        x = x + attn(norm_1(x), attn_mask=attn_mask)
        x = x + self.moe(norm_2(x), fwd_mask=fwd_mask, bck_mask=bck_mask)
        return x


class MoETransformer(Transformer):
    def __init__(
        self,
        config: RoutedModelConfig,
        labels: list[str],
    ) -> None:
        
        super().__init__(config)
        assert labels[0] == "core", "first label must be core"

        align = 64
        core_dim = round(config.core_param_prc * config.mlp_dim / align) * align
        aux_dim = round(config.aux_param_prc * config.mlp_dim / align) * align

        self.labels = labels
        self.embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.norm = nn.RMSNorm(config.embed_dim)
        self.unembed = nn.Linear(config.embed_dim, config.vocab_size, bias=True)
        
        self.blocks = []
        for _ in range(config.num_layers):
            self.blocks.append(Block(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                num_key_value=config.num_key_value,
                attn_bias=config.attn_bias,
                num_experts=len(labels),
                core_dim=core_dim,
                aux_dim=aux_dim,
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

        assert label in self.labels, f"label {label} not found in {self.labels}"

        if label == "core":
            yield from self.embed.parameters()
            yield from self.unembed.parameters()
            yield from self.norm.parameters()

        for block in self.blocks:

            if label == "core":
                yield from block.attn.parameters()
                yield from block.norm_1.parameters()
                yield from block.norm_2.parameters()

            e_idx = self.labels.index(label)
            yield from block.moe.experts[e_idx].parameters()


    def forward(
        self,
        tokens: torch.Tensor, #(B, T)
        targets: Optional[torch.Tensor], #(B, T)
        fwd_mask: torch.Tensor, #(K,)
        bck_mask: torch.Tensor, #(K,)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

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

        logits = run_stack(tokens, fwd_mask=fwd_mask, bck_mask=bck_mask)  # (B, T, V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1000,
                reduction="mean",
            )

        return logits, loss