"""Validate the three completed Phase 2 training runs.

Run with ``python -m analysis.stories_phase2.verify``. Explicit run directories
may be supplied when discovery is undesirable.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

from analysis.stories_phase2.common import (
    ALL_EXPERTS,
    AUX_LABELS,
    DATA_LABELS,
    EXPECTED_MODEL_PARAMS,
    EXPECTED_MODEL_SHAPE,
    EXPECTED_TOKEN_BUDGETS,
    PRIMARY_AUX_LABEL,
    is_finite_number,
    read_json,
    read_jsonl,
    retained_key,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results/stories_phase2"
STAGE_FOR_KIND = {"gram": "routed", "baseline": "baseline", "filtered": "filtering"}


def _stage_state(run_dir: Path, stage_name: str) -> dict[str, Any] | None:
    path = run_dir / stage_name / "stage.json"
    return read_json(path) if path.is_file() else None


def discover_run(results_root: Path, kind: str) -> Path:
    """Select the newest completed run matching ``kind``."""
    stage_name = STAGE_FOR_KIND[kind]
    candidates: list[Path] = []
    for config_path in results_root.glob("seed_*/*/config.json"):
        run_dir = config_path.parent
        try:
            config = read_json(config_path)
            names = [stage.get("name") for stage in config.get("stages", [])]
            state = _stage_state(run_dir, stage_name)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if names == [stage_name] and state and state.get("completed") is True:
            candidates.append(run_dir)
    if not candidates:
        raise FileNotFoundError(f"No completed {kind} run found below {results_root}")
    return max(candidates, key=lambda path: path.name)


def resolve_artifact_dir(run_dir: Path, kind: str, config: dict[str, Any]) -> Path:
    stage_name = STAGE_FOR_KIND[kind]
    stage_dir = run_dir / stage_name
    if kind != "filtered":
        return stage_dir
    retained = config["stages"][0].get("retain_targets")
    if not isinstance(retained, list) or len(retained) != 1:
        return stage_dir
    return stage_dir / "_".join(retained[0])


def _check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def _validate_config(kind: str, config: dict[str, Any], failures: list[str]) -> None:
    stage_name = STAGE_FOR_KIND[kind]
    run = config.get("run", {})
    stages = config.get("stages", [])
    _check(len(stages) == 1 and stages[0].get("name") == stage_name,
           f"expected one {stage_name} stage", failures)
    _check(run.get("seed") == 1, "seed must be 1", failures)
    _check(run.get("device") == "mps", "device must be mps", failures)
    _check(run.get("dtype") == "torch.float32", "dtype must be torch.float32", failures)
    _check(run.get("compile") is False, "compile must be disabled", failures)
    _check(run.get("micro_batch_size") == 16, "micro batch size must be 16", failures)
    _check(run.get("accumulation_steps") == 8, "accumulation steps must be 8", failures)
    _check(run.get("effective_batch_size") == 128, "effective batch size must be 128", failures)
    _check(run.get("nominal_token_budget") == EXPECTED_TOKEN_BUDGETS[stage_name],
           f"unexpected nominal token budget for {kind}", failures)
    _check(run.get("model_shape") == "paper", "model shape marker must be paper", failures)
    model = stages[0].get("model", {}) if kind == "gram" and stages else config.get("model", {})
    for key, expected in EXPECTED_MODEL_SHAPE.items():
        _check(model.get(key) == expected, f"model {key} must be {expected}", failures)
    if kind == "gram":
        _check(model.get("arch") == "moe", "GRAM architecture must be moe", failures)
        _check(model.get("core_param_prc") == 1.0, "GRAM core_param_prc must be 1.0", failures)
        _check(model.get("aux_param_prc") == 0.1, "GRAM aux_param_prc must be 0.1", failures)
        expected_profiles = [
            ["core"],
            *[["core", label] for label in AUX_LABELS],
            list(ALL_EXPERTS),
            [label for label in ALL_EXPERTS if label != PRIMARY_AUX_LABEL],
        ]
        actual = stages[0].get("retain_targets") if stages else None
        _check(
            actual is not None and {retained_key(x) for x in actual} == {retained_key(x) for x in expected_profiles},
            "GRAM retained-label profiles do not match Phase 2", failures,
        )
    elif kind == "filtered" and stages:
        expected = tuple(sorted(set(ALL_EXPERTS) - {PRIMARY_AUX_LABEL}))
        actual = stages[0].get("retain_targets")
        _check(actual is not None and len(actual) == 1 and retained_key(actual[0]) == expected,
               "filtered run retained labels are incorrect", failures)


def _validate_losses(path: Path, kind: str, final_step: int, failures: list[str]) -> None:
    try:
        with path.open("rb") as handle:
            losses = pickle.load(handle)
    except Exception as exc:
        failures.append(f"could not load loss history: {exc}")
        return
    _check(isinstance(losses, dict) and isinstance(losses.get("train"), dict),
           "loss history must contain a train mapping", failures)
    if not isinstance(losses, dict) or not isinstance(losses.get("train"), dict):
        return
    expected_nonempty = set(DATA_LABELS)
    if kind == "filtered":
        expected_nonempty.remove(PRIMARY_AUX_LABEL)
    observed_steps: list[float] = []
    for label in DATA_LABELS:
        values = np.asarray(losses["train"].get(label, []))
        _check(np.isfinite(values).all(), f"non-finite training loss for {label}", failures)
        if label in expected_nonempty:
            _check(values.ndim == 2 and values.shape[1] == 2 and len(values) > 0,
                   f"missing training losses for {label}", failures)
            if values.ndim == 2 and values.shape[1] == 2 and len(values):
                observed_steps.extend(values[:, 0].tolist())
        else:
            _check(values.size == 0, f"held-out label {label} has training losses", failures)
    _check(bool(observed_steps) and int(max(observed_steps)) == final_step - 1,
           "loss history does not reach the final optimizer iteration", failures)


def _validate_evaluations(path: Path, kind: str, failures: list[str]) -> tuple[int, list[dict[str, Any]]]:
    try:
        rows = [row for row in read_jsonl(path) if row.get("function") == "do_eval"]
    except (OSError, ValueError) as exc:
        failures.append(f"could not parse evaluation records: {exc}")
        return 0, []
    expected_count = 35 if kind == "gram" else 5
    _check(len(rows) == expected_count, f"expected {expected_count} evaluation records, found {len(rows)}", failures)
    keys: set[tuple[Any, Any]] = set()
    for row in rows:
        _check(row.get("data_label") in DATA_LABELS, "evaluation has an unexpected data label", failures)
        _check(is_finite_number(row.get("loss")), "evaluation has a non-finite loss", failures)
        key = (retained_key(row.get("expert_labels")), row.get("data_label"))
        _check(key not in keys, f"duplicate evaluation record: {key}", failures)
        keys.add(key)
    _check({row.get("data_label") for row in rows} == set(DATA_LABELS),
           "evaluation records do not cover all five labels", failures)
    return len(rows), rows


def validate_run(run_dir: Path, kind: str) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    failures: list[str] = []
    config_path = run_dir / "config.json"
    stats_path = run_dir / "stats.jsonl"
    stage_name = STAGE_FOR_KIND[kind]
    state_path = run_dir / stage_name / "stage.json"
    for path in (config_path, stats_path, state_path):
        _check(path.is_file(), f"missing artifact: {path}", failures)
    if not config_path.is_file():
        return {"kind": kind, "run_dir": str(run_dir), "passed": False, "failures": failures}
    try:
        config = read_json(config_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        failures.append(f"could not parse config: {exc}")
        return {"kind": kind, "run_dir": str(run_dir), "passed": False, "failures": failures}
    _validate_config(kind, config, failures)
    if state_path.is_file():
        try:
            state = read_json(state_path)
            _check(state.get("completed") is True, "completion marker is false", failures)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"could not parse completion marker: {exc}")
    artifact_dir = resolve_artifact_dir(run_dir, kind, config)
    checkpoint_path = artifact_dir / "checkpoint.pth"
    losses_path = artifact_dir / "losses.pkl"
    _check(checkpoint_path.is_file(), f"missing final checkpoint: {checkpoint_path}", failures)
    _check(losses_path.is_file(), f"missing loss history: {losses_path}", failures)
    final_step = -1
    total_steps = -1
    parameter_count = -1
    if checkpoint_path.is_file():
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
            final_step = int(checkpoint.get("step", -1))
            total_steps = int(checkpoint.get("total_steps", -1))
            model = checkpoint.get("model")
            _check(isinstance(model, dict) and bool(model), "checkpoint has no model state", failures)
            if isinstance(model, dict):
                parameter_count = sum(value.numel() for value in model.values())
                _check(parameter_count == EXPECTED_MODEL_PARAMS[stage_name],
                       f"unexpected checkpoint parameter count: {parameter_count}", failures)
            _check(final_step > 0 and final_step == total_steps,
                   f"checkpoint step {final_step} does not equal total {total_steps}", failures)
        except Exception as exc:
            failures.append(f"could not load final checkpoint: {exc}")
    if losses_path.is_file() and final_step > 0:
        _validate_losses(losses_path, kind, final_step, failures)
    evaluation_count, _ = _validate_evaluations(stats_path, kind, failures) if stats_path.is_file() else (0, [])
    return {
        "kind": kind,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "artifact_dir": str(artifact_dir.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "final_step": final_step,
        "total_steps": total_steps,
        "parameter_count": parameter_count,
        "evaluation_count": evaluation_count,
        "passed": not failures,
        "failures": failures,
    }


def verify_phase2(
    results_root: Path = DEFAULT_RESULTS_ROOT,
    gram_dir: Path | None = None,
    baseline_dir: Path | None = None,
    filtered_dir: Path | None = None,
) -> dict[str, Any]:
    supplied = {"gram": gram_dir, "baseline": baseline_dir, "filtered": filtered_dir}
    reports = {}
    for kind, path in supplied.items():
        reports[kind] = validate_run(path or discover_run(results_root, kind), kind)
    failures = [f"{kind}: {failure}" for kind, report in reports.items() for failure in report["failures"]]
    return {"passed": not failures, "failures": failures, "runs": reports}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--gram-dir", type=Path)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--filtered-dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = verify_phase2(args.results_root, args.gram_dir, args.baseline_dir, args.filtered_dir)
    output = args.output or args.results_root / "phase2_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
