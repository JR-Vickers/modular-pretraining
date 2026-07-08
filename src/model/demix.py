"""
DEMix domain-expert model.

Analogue of ``src/model/moe.py`` for the DEMix method
(Gururangan et al., 2021, arXiv:2108.05036): every feedforward layer is
replaced by a collection of *domain experts*, one per data label. Routing is
hard and observable (by domain label, at the sequence level) rather than
learned — a sequence is sent to its single domain expert. Unlike the MoE here,
where small aux experts *add* to an always-on core, each DEMix expert is a
full-size FFN that *replaces* the feedforward block for its domain.

Everything that is not a domain expert — token embeddings, the final norm,
the unembedding, and the per-block attention + RMSNorms — is **shared** across
all domains and trained on every batch (it is the "SHARED" parameter group).

The ``freeze`` mechanism is imported verbatim from ``src.model.moe`` so the
control over what accumulates gradient is identical to the MoE model and stays
compatible with heterogeneous accumulation windows (windows whose micro-batches
route to different experts): within a window each expert only receives gradient
from its own micro-batches, the shared trunk receives gradient from any
micro-batch that trains *something*, and the training loop steps exactly the
optimizers whose parameters got a gradient.

At inference, ``fwd_mask`` can select more than one expert; their outputs are
uniformly averaged — the parameter-free analogue of DEMix expert "mixing".
"""

from typing import Generator, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import RoutedModelConfig, Transformer
from src.model.base import make_attention_mask, Attention, MLP
from src.model.moe import freeze  # identical freeze semantics to the MoE model


class DemixLayer(nn.Module):
    """Domain-expert feedforward layer.

    One full-size MLP expert per label. A sequence routes to its single domain
    expert (``fwd_mask`` has exactly one active entry during training). The
    expert *replaces* the feedforward block — there is no always-on core that
    is summed in (contrast ``src.model.moe.MoE``).
    """

    def __init__(
        self,
        embed_dim: int,
        num_experts: int,
        mlp_dim: int,
    ) -> None:

        super().__init__()

        # Every expert is a full-size FFN (DEMix: an expert replaces the FFN).
        self.experts = nn.ModuleList(
            [MLP(embed_dim, mlp_dim) for _ in range(num_experts)]
        )

    def forward(
        self,
        x: torch.Tensor,
        fwd_mask: torch.Tensor,
        bck_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        x:        (B, T, E)
        fwd_mask: (K,) boolean — which experts run.
        bck_mask: (K,) boolean — which of those accumulate gradient
                  (others run frozen via functional_call).
        returns:  (B, T, E)

        Single active expert (training / single-domain inference) -> that
        expert's output. Multiple active experts (inference "mixing") -> the
        uniform average of their outputs.
        """
        y = None
        num_active = 0
        for i in range(len(self.experts)):
            if fwd_mask[i]:
                expert = freeze(self.experts[i], do_freeze=not bck_mask[i])
                out = expert(x)
                y = out if y is None else y + out
                num_active += 1

        # Uniform ensemble across selected experts (no-op when exactly one).
        if num_active > 1:
            y = y / num_active

        return y


class Block(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_key_value: int,
        attn_bias: bool,
        num_experts: int,
        mlp_dim: int,
    ) -> None:

        super().__init__()

        self.attn = Attention(
            num_heads=num_heads,
            embed_dim=embed_dim,
            num_key_value=num_key_value,
            bias=attn_bias,
        )

        self.demix = DemixLayer(
            embed_dim=embed_dim,
            num_experts=num_experts,
            mlp_dim=mlp_dim,
        )

        self.norm_1 = nn.RMSNorm(embed_dim)
        self.norm_2 = nn.RMSNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        fwd_mask: torch.Tensor,  # (K,)
        bck_mask: torch.Tensor,  # (K,)
    ) -> torch.Tensor:

        # Shared trunk trains whenever *any* expert is being trained this batch.
        freeze_shared = not bool(bck_mask.any())
        attn = freeze(self.attn, do_freeze=freeze_shared)
        norm_1 = freeze(self.norm_1, do_freeze=freeze_shared)
        norm_2 = freeze(self.norm_2, do_freeze=freeze_shared)

        x = x + attn(norm_1(x), attn_mask=attn_mask)
        x = x + self.demix(norm_2(x), fwd_mask=fwd_mask, bck_mask=bck_mask)
        return x


class DemixTransformer(Transformer):
    """DEMix transformer: shared trunk + per-domain full-size FFN experts."""

    def __init__(
        self,
        config: RoutedModelConfig,
        labels: list[str],
    ) -> None:

        super().__init__(config)
        assert labels[0] == "core", "first label must be core"

        self.labels = labels

        self.embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.norm = nn.RMSNorm(config.embed_dim)
        self.unembed = nn.Linear(config.embed_dim, config.vocab_size, bias=True)

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim=config.embed_dim,
                    num_heads=config.num_heads,
                    num_key_value=config.num_key_value,
                    attn_bias=config.attn_bias,
                    num_experts=len(labels),
                    mlp_dim=config.mlp_dim,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if hasattr(module, "bias") and getattr(module, "bias") is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.RMSNorm):
            module.weight.data.fill_(1.0)

    def get_params(self, label: str) -> Generator[torch.Tensor, None, None]:
        """Parameters owned by a label.

        ``"SHARED"`` -> trunk (embed, final norm, unembed, and per-block
        attention + RMSNorms). A domain label -> only that domain's expert
        MLPs. This separation (unlike the MoE model, where the trunk is owned
        by ``"core"``) is what lets the trunk be stepped on every domain while
        each expert is stepped only on its own data.
        """
        all_labels = self.labels + ["SHARED"]
        assert label in all_labels, f"label {label} not found in {all_labels}"

        if label == "SHARED":
            yield from self.embed.parameters()
            yield from self.norm.parameters()
            yield from self.unembed.parameters()
            for block in self.blocks:
                yield from block.attn.parameters()
                yield from block.norm_1.parameters()
                yield from block.norm_2.parameters()
            return

        e_idx = self.labels.index(label)
        for block in self.blocks:
            yield from block.demix.experts[e_idx].parameters()

    def forward(
        self,
        tokens: torch.Tensor,  # (B, T)
        targets: Optional[torch.Tensor],  # (B, T)
        fwd_mask: torch.Tensor,  # (K,)
        bck_mask: torch.Tensor,  # (K,)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # Shared trunk trains whenever any expert trains this batch.
        freeze_shared = not bool(bck_mask.any())
        embed = freeze(self.embed, do_freeze=freeze_shared)
        norm = freeze(self.norm, do_freeze=freeze_shared)
        unembed = freeze(self.unembed, do_freeze=freeze_shared)

        def run_stack(
            tok: torch.Tensor,
            fwd_mask: torch.Tensor,
            bck_mask: torch.Tensor,
        ) -> torch.Tensor:

            attn_mask = make_attention_mask(tok, self.config.eos_token_id)
            h = embed(tok)
            for block in self.blocks:
                h = block(
                    h,
                    attn_mask=attn_mask,
                    fwd_mask=fwd_mask,
                    bck_mask=bck_mask,
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
