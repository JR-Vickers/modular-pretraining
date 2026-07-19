from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable


PRIMARY_AUX_LABEL = "a-deadline-or-time-limit"
AUX_LABELS = (
    PRIMARY_AUX_LABEL,
    "alien-encounters",
    "bygone-eras",
    "cultural-traditions",
)
DATA_LABELS = ("core", *AUX_LABELS)
ALL_EXPERTS = DATA_LABELS
EXPECTED_TOKEN_BUDGETS = {
    "routed": 547_853_673,
    "baseline": 547_853_673,
    "filtering": 536_228_665,
}
EXPECTED_MODEL_PARAMS = {
    "routed": 32_571_904,
    "baseline": 26_257_920,
    "filtering": 26_257_920,
}
EXPECTED_MODEL_SHAPE = {
    "ctx_len": 256,
    "vocab_size": 4096,
    "num_layers": 8,
    "num_heads": 8,
    "num_key_value": 2,
    "embed_dim": 512,
    "mlp_dim": 2048,
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected an object on {path}:{line_number}")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def profile_id(omitted_label: str | None) -> str:
    return "all_on" if omitted_label is None else f"leave_out__{omitted_label}"


def evaluation_profiles() -> list[dict[str, Any]]:
    profiles = [{
        "profile_id": profile_id(None),
        "omitted_label": None,
        "expert_labels": list(ALL_EXPERTS),
    }]
    profiles.extend(
        {
            "profile_id": profile_id(label),
            "omitted_label": label,
            "expert_labels": [item for item in ALL_EXPERTS if item != label],
        }
        for label in AUX_LABELS
    )
    return profiles


def retained_key(labels: Iterable[str] | None) -> tuple[str, ...] | None:
    return None if labels is None else tuple(sorted(labels))
