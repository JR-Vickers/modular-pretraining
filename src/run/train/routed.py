"""
Gradient-routing training loops
"""

import contextlib
import pickle
import random
import sys
import torch
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm
from dataclasses import dataclass, field
from collections import Counter

from src.model.config import Transformer, ModelConfig, RoutedModelConfig
from src.model.moe import MoETransformer
from src.model.lora import LoRATransformer
from src.run.eval import eval_loss
from src.run.util.config import ExperimentConfig, StageConfig, use_fused_adamw
from src.run.util.distributed import get_raw_model, barrier, broadcast_object, is_main_process, get_rank
from src.run.util.logger import get_tqdm_kwargs
from src.run.util.preemption import is_preempted
from src.run.util.tools import get_batch, get_exp_mask, log_batch_counts, set_seeds
from src.run.util.state import save_checkpoint, should_save


# --------------------------------------------------------------------------- #
# Routed stage configs                                                        #
# --------------------------------------------------------------------------- #


@dataclass
class RoutedConfig(StageConfig):
    model: ModelConfig = field(default_factory=RoutedModelConfig)
    name: str = "routed"
    eval_arbsub: bool = False
    label_prc: float = 1.0


@dataclass
class OrderedConfig(RoutedConfig):
    """Two-phase ordered routing (base then fine-tune)."""
    aux_factor: float | dict[str, float] = 1.0
    core_aux_ratio: float = 1.0
    equal_compute: bool = True


@dataclass
class UnorderedConfig(RoutedConfig):
    """Single-phase unordered routing with robust batches."""
    aux_factor: float | dict[str, float] = 1.0
    robust_prc: float = 0.5
    aux_route_prc: float = 0.0
    equal_compute: bool = True


def do_routed(
    stage: RoutedConfig,
    model: Transformer,
    config: ExperimentConfig,
    batch_groups: list[list[tuple[str, tuple, tuple]]],
    state: dict | None = None,
) -> Transformer:
    """Core gradient-routing training loop.

    Iterates over batch_groups sequentially.  Within each group, batches are
    processed with per-expert optimizer stepping — only the experts listed in
    ``bck_experts`` have their optimizer stepped, while all experts in
    ``fwd_experts`` contribute to the forward pass (controlled via select_mask).

    Each batch group gets its own warmup/constant/decay LR cycle, with phase
    lengths proportional to that group's batch count.

    Returns:
        Trained model
    """

    # unpack run config
    acc_steps = config.run.accumulation_steps
    loaders = config.run.loaders
    epochs = config.run.epochs
    logger = config.run.logger
    adam_betas = config.run.adam_betas
    data_labels = config.run.labels
    aux_labels = config.data.aux.labels
    seed = config.run.seed
    warmup_prc = config.run.warmup_prc
    decay_prc = config.run.decay_prc
    is_ddp = config.run.is_ddp
    all_labels = config.run.labels

    #unpack stage config
    lr = stage.lr
    num_evals = stage.num_train_evals
    num_checkpoints = stage.num_checkpoints
    res_dir = stage.res_dir
    acc_mode = stage.acc_mode

    logger.info(f"---- Begin Routed ----")
    logger.debug(f"acc_mode: {acc_mode}")

    batch_groups = broadcast_object(batch_groups, src=0)

    if acc_mode == "uniform":
        for i, bg in enumerate(batch_groups):
            counter = Counter(bg)
            temp = []
            for value, count in counter.items():
                temp += [value] * round(count / acc_steps)
            random.seed(seed)
            random.shuffle(temp)
            batches = []
            for value in temp:
                batches += [value] * acc_steps
            batch_groups[i] = broadcast_object(batches, src=0)

    else: #heterogeneous
        for i, bg in enumerate(batch_groups):
            random.seed(get_rank())
            random.shuffle(bg) #different per rank
            if len(bg) % acc_steps != 0: #if not multiple of acc_steps, truncate
                bg = bg[:-(len(bg) % acc_steps)]
            batch_groups[i] = bg

    set_seeds(seed)

    model.train()
    raw_model = get_raw_model(model)

    # validate batches
    for bg in batch_groups:
        assert len(bg) % acc_steps == 0
        for l_name, fwd_experts, bck_experts in bg:
            assert l_name in data_labels
            assert all(e in data_labels for e in fwd_experts)
            assert all(e in data_labels for e in bck_experts)

    assert lr > 0, "LR must be provided"

    label_to_idx = {label: i for i, label in enumerate(data_labels)}

    if state is None:
        state = dict()

    # calculate total steps (optimizer steps across all groups)
    num_total_steps = sum(len(bg) // acc_steps for bg in batch_groups) * epochs
    logger.info(f"num_total_steps: {num_total_steps}")
    eval_freq = max(1, num_total_steps // num_evals) if num_evals > 0 else -1
    checkpoint_freq = max(1, num_total_steps // num_checkpoints) if num_checkpoints > 0 else -1
    resume_step = state.get("step", -1)

    # setup optimizers (one per label — selective stepping induces modularity)
    opts = {}
    for label in data_labels:
        params = list(raw_model.get_params(label))
        opts[label] = torch.optim.AdamW(
            params, lr=lr, fused=use_fused_adamw(config.run.device), betas=adam_betas
        )

    # restore optimizers
    if "opts" in state:
        for label in opts.keys():
            if label in state["opts"]:
                opts[label].load_state_dict(state["opts"][label])
                opts[label].param_groups[0]['lr'] = lr
            else:
                logger.warning(f"Label {label} not found in state['opts']")
        del state["opts"]

    # setup losses
    losses = {"train": {}, "val": {}}
    for label in data_labels:
        losses["train"][label] = []
        losses["val"][label] = []

    # restore losses (each entry is a (step, loss) tuple)
    if "losses" in state:
        for label in losses["train"].keys():
            losses["train"][label] = [tuple(x) for x in state["losses"]["train"].get(label, [])]
        for label in losses["val"].keys():
            losses["val"][label] = [tuple(x) for x in state["losses"]["val"].get(label, [])]

        logger.info(f"Restored {sum(len(v) for v in losses['train'].values())} train loss entries")
        logger.info(f"Restored {sum(len(v) for v in losses['val'].values())} val loss entries")

    # setup scheduler
    # Precompute per-group schedule boundaries (in optimizer steps).
    # Each group gets its own warmup→constant→decay cycle.
    end_factor = 1e-8 / lr
    group_schedules = []
    cumulative = 0
    for gi, bg in enumerate(batch_groups):
        group_len = (len(bg) // acc_steps) * epochs #num steps in group
        w = max(1, round(warmup_prc * group_len))
        d = max(1, round(decay_prc * group_len))
        c = group_len - w - d
        group_schedules.append({
            "start": cumulative,
            "len": group_len,
            "warmup": w,
            "constant": c,
            "decay": d,
        })
        cumulative += group_len
        logger.info(
            f"LR schedule group {gi}: steps [{group_schedules[-1]['start']}, "
            f"{cumulative}), warmup={w}, constant={c}, decay={d}"
        )

    def lr_lambda(current_step):
        g = group_schedules[-1]
        for gs in group_schedules:
            if current_step < gs["start"] + gs["len"]:
                g = gs
                break

        local_step = current_step - g["start"]
        w, c, d = g["warmup"], g["constant"], g["decay"]

        if local_step < w:
            if local_step == 0:
                logger.info(f"LR Scheduler: Group warmup (step {current_step})")
            return end_factor + (1.0 - end_factor) * (local_step / w)
        elif local_step < w + c:
            if local_step == w:
                logger.info(f"LR Scheduler: Group constant (step {current_step})")
            return 1.0
        else:
            if local_step == w + c:
                logger.info(f"LR Scheduler: Group decay (step {current_step})")
            progress = (local_step - w - c) / d
            return 1.0 - (1.0 - end_factor) * progress

    scheduler = LambdaLR(opts["core"], lr_lambda)
    cur_lr = scheduler.get_last_lr()[0]
    for label in opts.keys():
        opts[label].param_groups[0]['lr'] = cur_lr

    # restore scheduler position from step count
    if resume_step > 0:
        scheduler.last_epoch = resume_step
        cur_lr = scheduler.get_lr()[0]
        scheduler._last_lr = [cur_lr]
        for label in opts.keys():
            opts[label].param_groups[0]['lr'] = cur_lr
        logger.info(f"Restored scheduler to step {resume_step}, LR: {cur_lr:.4e}")

    # memory diagnostics (before first training step)
    if is_main_process() and config.run.device.type == "cuda":
        mem_summary = torch.cuda.memory_summary()
        logger.info(f"CUDA memory before training loop:")
        for line in mem_summary.split('\n'):
            logger.info(line)

    # training loop
    step = 0
    for batches in batch_groups:

        group_epoch_steps = len(batches) // acc_steps
        group_total_steps = group_epoch_steps * epochs
        pbar = tqdm(total=group_total_steps, **get_tqdm_kwargs(logger, ncols=150))

        logger.info(f"Batch group start, LR: {cur_lr:.4e}, batches: {len(batches)}, epochs: {epochs}, group_total_steps: {group_total_steps}")

        for epoch_idx in range(epochs):

            for label in data_labels:
                loaders[label]["train"].reset(epoch_idx)

            for batch_idx in range(len(batches)):

                batch = batches[batch_idx]
                loader_name, experts_forward, experts_backward = batch
                loader = loaders[loader_name]["train"]
                x, y, batch_label = get_batch(loader)

                if step < resume_step:
                    step += 1
                    continue

                fwd_mask = get_exp_mask(data_labels, experts_forward, device=x.device)
                bck_mask = get_exp_mask(data_labels, experts_backward, device=x.device)

                acc_idx = batch_idx % acc_steps
                is_last_acc = acc_idx == acc_steps - 1
                no_sync_ctx = model.no_sync() if (is_ddp and not is_last_acc) else contextlib.nullcontext()

                with no_sync_ctx:

                    _, loss = model.forward(
                        tokens=x,
                        targets=y,
                        fwd_mask=fwd_mask,
                        bck_mask=bck_mask,
                    )

                    scaled_loss = loss / acc_steps
                    scaled_loss.backward()

                # train loss logging
                cur_loss = loss.item()
                losses["train"][batch_label].append((step, cur_loss))

                # update progress bar
                num_subsets = len(all_labels)
                exp_for_str = ','.join([str(label_to_idx[e]) for e in experts_forward])
                exp_for_str = exp_for_str.ljust(num_subsets * 2 - 1)
                exp_bck_str = ','.join([str(label_to_idx[e]) for e in experts_backward])
                exp_bck_str = exp_bck_str.ljust((num_subsets + 1)* 2 - 1)
                label_idx = data_labels.index(batch_label)
                desc_str = f"AC {acc_idx + 1}/{acc_steps} | LR: {cur_lr:.2e} | L: {cur_loss:.2f} | LB: {label_idx} | EF: {exp_for_str} | EB: {exp_bck_str}"

                pbar.set_description(desc_str)
                pbar.refresh()

                if is_last_acc:

                    step += 1
                    pbar.update()

                    # collect experts that received gradient this window
                    exp_to_step = [
                        label for label in data_labels
                        if any(p.grad is not None for p in raw_model.get_params(label))
                    ]

                    # clip the union
                    params_to_clip = [p for e in exp_to_step for p in raw_model.get_params(e)]
                    if params_to_clip:
                        torch.nn.utils.clip_grad_norm_(params_to_clip, 1.0)

                    # step only the experts that actually got gradient
                    for exp in exp_to_step:
                        opts[exp].step()

                    for opt in opts.values():
                        opt.zero_grad(set_to_none=True)

                    # update learning rate
                    scheduler.step()
                    cur_lr = scheduler.get_last_lr()[0]
                    for label in opts.keys():
                        opts[label].param_groups[0]['lr'] = cur_lr

                    # measure validation loss
                    if eval_freq > 0 and step % eval_freq == 0:
                        model.eval()
                        for label in data_labels:
                            eval_experts = ("core", label) if label in aux_labels else ("core",)
                            val_loss = eval_loss(
                                model, config,
                                data_label=label,
                                num_batches=50,
                                expert_labels=eval_experts,
                                shuffle_seed=step,
                            )
                            losses["val"][label].append((step, val_loss))
                        model.train()

                    # memory diagnostics (after first step)
                    if (
                        step == resume_step + 2
                        and is_main_process()
                        and config.run.device.type == "cuda"
                    ):
                        mem_summary = torch.cuda.memory_summary()
                        logger.info(f"CUDA memory after first training step:")
                        for line in mem_summary.split('\n'):
                            logger.info(line)

                    # logger printout
                    if (step == 1) or (step % 1000 == 0) or (step == num_total_steps):

                        train_loss_str = ""
                        for label in losses["train"].keys():
                            label_str = label.upper()
                            loss_slice = [v for _, v in losses["train"][label][-100:]]
                            avg_train_loss = np.mean(loss_slice) if len(loss_slice) > 0 else float('nan')
                            train_loss_str += f"{label_str}: {avg_train_loss:.2f} "
                        logger.info(f"Step: {step}, LR: {cur_lr:.4e}")
                        logger.info(f"Train Loss: {train_loss_str}")

                        if eval_freq > 0:
                            val_loss_str = ""
                            for label in losses["val"].keys():
                                label_str = label.upper()
                                val_loss = losses["val"][label][-1][1] if len(losses["val"][label]) > 0 else float('nan')
                                val_loss_str += f"{label_str}: {val_loss:.2f} "
                            logger.info(f"Val Loss: {val_loss_str}")

                    # save checkpoint periodically or at final step
                    if should_save(step, num_total_steps, checkpoint_freq):

                        ckpt_state = {
                            'opts': {label: opt.state_dict() for label, opt in opts.items()},
                            'step': step,
                            'total_steps': num_total_steps,
                            'losses': {
                                "train": {label: np.array(vals) for label, vals in losses["train"].items()},
                                "val": {label: np.array(vals) for label, vals in losses["val"].items()},
                            },
                        }

                        save_checkpoint(
                            stage=stage,
                            model=model,
                            state=ckpt_state,
                            config=config,
                        )

                        if is_preempted():
                            logger.warning("Preemption detected — checkpoint saved, exiting")
                            sys.exit(0)

                        if is_main_process() and step == num_total_steps:
                            res_dir.mkdir(parents=True, exist_ok=True)
                            losses_path = res_dir / "losses.pkl"
                            losses_path.write_bytes(pickle.dumps(ckpt_state["losses"]))

            # sync after epoch
            barrier()

        # close progress bar
        pbar.close()

    return model

def do_routed_ordered(
    stage: OrderedConfig,
    model: Transformer,
    config: ExperimentConfig,
    do_ft: bool = True,
    state: dict | None = None
) -> Transformer:
    """
    Two-phase ordered gradient routing.
    """

    # unpack config
    loaders = config.run.loaders
    logger = config.run.logger
    all_labels = config.run.labels
    aux_labels = config.data.aux.labels
    seed = config.run.seed
    aux_factor = stage.aux_factor # what percent of the aux data do you use
    core_aux_ratio = stage.core_aux_ratio # what percent of the ft phase is made of core data
    equal_compute = stage.equal_compute
    label_prc = stage.label_prc

    logger.info("---- Begin Routed Ordered ----")

    raw_model = get_raw_model(model)
    assert isinstance(raw_model, MoETransformer) or isinstance(raw_model, LoRATransformer), "raw_model must be a MoETransformer or LoRATransformer"

    set_seeds(seed)

    if isinstance(aux_factor, float):
        factor = aux_factor
        aux_factor = {label: factor for label in aux_labels}

    assert all(val > 0.0 for val in aux_factor.values()), f"all aux_factor values must be > 0.0"
    assert 0 <= core_aux_ratio, f"0 <= core_aux_ratio"
    assert 0 <= label_prc <= 1, f"0 <= label_prc <= 1"

    batch_groups: list[list[tuple[str, tuple, tuple]]] = []

    len_all = len(loaders["all"]["train"])
    len_all_labeled = round(len_all * label_prc)
    len_all_unlabeled = len_all - len_all_labeled

    len_aux = 0
    for label, factor in aux_factor.items():
        len_aux += round(len(loaders[label]["train"]) * label_prc * factor)
    assert len_aux <= len_all_labeled, f"len_aux <= len_all_labeled, ({len_aux} <= {len_all_labeled})"

    len_aux_for_ft = len_aux
    len_core_for_ft = round(len_aux_for_ft * core_aux_ratio)
    len_ft = len_aux_for_ft + len_core_for_ft
    len_base = len_all_labeled - len_ft
    prc_core_ft = (len_core_for_ft / len_ft) if len_ft > 0 else 0.0

    logger.info(f"initial len_all: {len_all}, len_all_labeled: {len_all_labeled}, len_all_unlabeled: {len_all_unlabeled}, len_ft: {len_ft}, len_base: {len_base}, label_prc: {label_prc}")

    if equal_compute:

        core_params = sum(p.numel() for p in raw_model.get_params("core"))
        total_aux_params = 0
        for label in aux_labels:
            total_aux_params += sum(p.numel() for p in raw_model.get_params(label))
        avg_aux_params = total_aux_params / len(aux_labels)

        baseline_flops = len_all * config.run.num_base_params
        base_flops = len_base * core_params
        ft_flops = len_ft * (core_params + avg_aux_params)
        unlabeled_flops = len_all_unlabeled * core_params #just core params bc we add to base
        routed_flops = base_flops + ft_flops + unlabeled_flops

        flop_diff = baseline_flops - routed_flops
        if flop_diff > 0:
            logger.info(f"compute-equal: {flop_diff} > 0, must extend training")
        else:
            logger.info(f"compute-equal: {flop_diff} <= 0, must truncate training")

        prc_base = base_flops / routed_flops
        prc_ft = ft_flops / routed_flops
        prc_unlabeled = unlabeled_flops / routed_flops

        diff_base = round( prc_base * flop_diff / core_params)
        diff_ft = round( prc_ft * flop_diff / (core_params + avg_aux_params))
        diff_unlabeled = round( prc_unlabeled * flop_diff / core_params)

        len_base = len_base + diff_base
        len_ft = len_ft + diff_ft
        len_all_unlabeled = len_all_unlabeled + diff_unlabeled

        logger.info(f"compute-equal: len_base: {diff_base} -> {len_base}, len_ft: {diff_ft} -> {len_ft}, len_all_unlabeled: {diff_unlabeled} -> {len_all_unlabeled}")

    len_core_for_ft = round(len_ft * prc_core_ft)
    len_aux_for_ft = len_ft - len_core_for_ft

    base_batches = []
    ft_batches = []
    unlabeled_batches = []

    base_batches += [("core", ("core",), ("core",))] * len_base

    aux_proportions = {label: len(loaders[label]["train"]) * aux_factor[label] for label in aux_labels}
    aux_proportions = {k: v / sum(aux_proportions.values()) for k, v in aux_proportions.items()}

    for label in aux_labels:

        cur_aux_prc = aux_proportions[label]
        cur_aux_ft_samples = round(len_aux_for_ft * cur_aux_prc)
        cur_core_ft_samples = round(len_core_for_ft * cur_aux_prc)

        cur_ft_batches = []
        cur_ft_batches += [(label, ("core", label), (label,))]  * cur_aux_ft_samples
        cur_ft_batches += [("core", ("core", label), (label,))] * cur_core_ft_samples

        random.shuffle(cur_ft_batches)
        cur_ft_batches = broadcast_object(cur_ft_batches, src=0)
        ft_batches.append(cur_ft_batches)

    unlabeled_batches = []
    for label in all_labels:
        proportion = len(loaders[label]["train"]) / len_all
        unlabeled_batches += [(label, ("core",), ("core",))] * round(proportion * len_all_unlabeled)

    logger.info(
        f"\nlen_all: {len_all}, len_all_labeled: {len_all_labeled}, len_all_unlabeled: {len_all_unlabeled},"
        f"\nlen_base: {len_base}, len_ft: {len_ft},"
        f"\nlen_core_for_ft: {len_core_for_ft}, len_aux_for_ft: {len_aux_for_ft},"
        f"\naux_factor: { {k: round(v, 4) for k, v in aux_factor.items()} }, core_aux_ratio: {round(core_aux_ratio, 4)},"
        f"\nequal_compute: {equal_compute}"
    )

    base_batches = base_batches + unlabeled_batches #for ordered, we just put unlabeled into the base phase

    count_by_label = {label: 0 for label in all_labels}
    for batch in base_batches:
        count_by_label[batch[0]] += 1
    for batch_group in ft_batches:
        for batch in batch_group:
            count_by_label[batch[0]] += 1
    logger.info(f"count_by_label: {count_by_label}")

    batch_groups = [base_batches]
    if do_ft and len(ft_batches) > 0:
        batch_groups.extend(ft_batches)

    len_batches = sum(len(x) for x in batch_groups)
    expected = len_base + len_ft + len_all_unlabeled
    if len_batches != expected:
        diff = expected - len_batches
        logger.warning(f"len_batches != expected, ({len_batches} != {expected}), {diff} length difference")

    for batch_group in batch_groups:
        log_batch_counts(batch_group, logger)

    # --- train ---

    return do_routed(
        stage=stage,
        model=model,
        config=config,
        batch_groups=batch_groups,
        state=state,
    )


def do_routed_unordered(
    stage: OrderedConfig,
    model: Transformer,
    config: ExperimentConfig,
    state: dict | None = None
) -> Transformer:
    """
    Single-phase unordered gradient routing.
    """

    # unpack run config
    loaders = config.run.loaders
    logger = config.run.logger
    all_labels = config.run.labels
    aux_labels = config.data.aux.labels
    aux_factor = stage.aux_factor
    robust_prc = stage.robust_prc
    aux_route_prc = stage.aux_route_prc
    seed = config.run.seed
    equal_compute = stage.equal_compute
    label_prc = stage.label_prc

    logger.info("---- Begin Routed Unordered ----")

    raw_model = get_raw_model(model)
    assert isinstance(raw_model, MoETransformer) or isinstance(raw_model, LoRATransformer), "raw_model must be a MoETransformer or LoRATransformer"

    set_seeds(seed)

    if isinstance(aux_factor, float):
        factor = aux_factor
        aux_factor = {label: factor for label in aux_labels}

    assert all(val > 0.0 for val in aux_factor.values()), f"all aux_factor values must be > 0.0"
    assert 0 <= robust_prc <= 1, f"0 <= robust_prc <= 1"
    assert 0 <= aux_route_prc <= 1, f"0 <= aux_route_prc <= 1"
    assert 0 <= label_prc <= 1, f"0 <= label_prc <= 1"

    batches: list[tuple[str, tuple, tuple]] = [] # [( label, (params_forward), (params_backward) ), ...]

    len_all = len(loaders["all"]["train"])
    len_all_labeled = round(len_all * label_prc)
    len_all_unlabeled = len_all - len_all_labeled

    len_aux = 0
    for label, factor in aux_factor.items():
        len_aux += round(len(loaders[label]["train"]) * label_prc * factor)
    assert len_aux <= len_all_labeled, f"len_aux <= len_all_labeled, ({len_aux} <= {len_all_labeled})"

    len_aux_for_ft = len_aux
    len_core = len_all_labeled - len_aux
    len_core_for_ft = round(len_core * robust_prc)
    len_core_for_base = len_core - len_core_for_ft
    len_ft = len_aux_for_ft + len_core_for_ft
    len_base = len_core_for_base
    prc_core_ft = (len_core_for_ft / len_ft) if len_ft > 0 else 0.0

    logger.info(f"initial len_all: {len_all}, len_all_labeled: {len_all_labeled}, len_all_unlabeled: {len_all_unlabeled}, len_ft: {len_ft}, len_base: {len_base}, label_prc: {label_prc}")

    if equal_compute:

        core_params = sum(p.numel() for p in raw_model.get_params("core"))
        total_aux_params = 0
        for label in aux_labels:
            total_aux_params += sum(p.numel() for p in raw_model.get_params(label))
        avg_aux_params = total_aux_params / len(aux_labels)

        # a value proportional to flops...
        baseline_flops = len_all * config.run.num_base_params
        base_flops = len_base * core_params
        ft_flops = len_ft * (core_params + avg_aux_params)
        unlabeled_flops = len_all_unlabeled * (core_params + total_aux_params)
        routed_flops = base_flops + ft_flops + unlabeled_flops

        flop_diff = baseline_flops - routed_flops
        if flop_diff > 0:
            logger.info(f"compute-equal: {flop_diff} > 0, must extend training")
        else:
            logger.info(f"compute-equal: {flop_diff} <= 0, must truncate training")

        prc_base = base_flops / routed_flops
        prc_ft = ft_flops / routed_flops
        prc_unlabeled = unlabeled_flops / routed_flops

        diff_base = round( prc_base * flop_diff / core_params)
        diff_ft = round( prc_ft * flop_diff / (core_params + avg_aux_params))
        diff_unlabeled = round( prc_unlabeled * flop_diff / (core_params + total_aux_params))

        len_base = len_base + diff_base
        len_ft = len_ft + diff_ft
        len_all_unlabeled = len_all_unlabeled + diff_unlabeled

        logger.info(f"compute-equal: len_base: {diff_base} -> {len_base}, len_ft: {diff_ft} -> {len_ft}, len_all_unlabeled: {diff_unlabeled} -> {len_all_unlabeled}")

    len_core_for_ft = round(len_ft * prc_core_ft)
    len_aux_for_ft = len_ft - len_core_for_ft

    base_batches = []
    ft_batches = []
    unlabeled_batches = []

    base_batches += [("core", ("core",), ("core",))] * len_base

    aux_proportions = {label: len(loaders[label]["train"]) * aux_factor[label] for label in aux_labels}
    aux_proportions = {k: v / sum(aux_proportions.values()) for k, v in aux_proportions.items()}

    for label in aux_labels:

        #--- calculate subset batch lengths (per-micro-batch) --

        cur_aux_prc = aux_proportions[label]
        cur_aux_ft_samples = round(len_aux_for_ft * cur_aux_prc)
        cur_core_ft_samples = round(len_core_for_ft * cur_aux_prc)

        aux_isrouted = round(cur_aux_ft_samples * aux_route_prc / aux_factor[label])
        aux_norouted = cur_aux_ft_samples - aux_isrouted

        ft_batches += [(label, ("core", label), ("core", label))] * aux_isrouted
        ft_batches += [(label, ("core", label), (label,))] * aux_norouted
        ft_batches += [("core", ("core", label), ("core", label))] * cur_core_ft_samples

    unlabeled_batches = []
    for label in all_labels:
        proportion = len(loaders[label]["train"]) / len_all
        unlabeled_batches += [(label, tuple(all_labels), tuple(all_labels))] * round(proportion * len_all_unlabeled)

    log_batch_counts(base_batches, logger)
    log_batch_counts(ft_batches, logger)
    log_batch_counts(unlabeled_batches, logger)

    logger.info(
        f"\nlen_all: {len_all}, len_all_labeled: {len_all_labeled}, len_all_unlabeled: {len_all_unlabeled},"
        f"\nlen_base: {len_base}, len_ft: {len_ft},"
        f"\nlen_core_for_ft: {len_core_for_ft}, len_aux_for_ft: {len_aux_for_ft},"
        f"\naux_factor: {aux_factor}, robust_prc: {round(robust_prc, 4)},"
        f"\naux_route_prc: {round(aux_route_prc, 4)}, equal_compute: {equal_compute}"
    )

    # --- combine ---

    batches = base_batches + ft_batches + unlabeled_batches

    count_by_label = {label: 0 for label in all_labels}
    for batch in batches:
        count_by_label[batch[0]] += 1
    logger.info(f"count_by_label: {count_by_label}")

    expected = len_base + len_ft + len_all_unlabeled
    if len(batches) != expected:
        diff = expected - len(batches)
        logger.warning(f"len(batches) != expected, ({len(batches)} != {expected}), {diff} length difference")

    log_batch_counts(batches, logger)

    # --- train ---

    return do_routed(
        stage=stage,
        model=model,
        config=config,
        batch_groups=[batches],
        state=state,
    )
