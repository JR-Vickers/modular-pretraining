from pathlib import Path
import re
import json

ROOT_DIR = Path(__file__).resolve().parents[3] / "results"

def parse_model_size(model_size: str) -> int:
    """Parse model size into parameter count.

    Supported forms:
      - Suffix strings: "50M", "2B", etc
    """

    text = str(model_size).strip().upper()
    match = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*([MB]?)", text)

    magnitude = float(match.group(1))
    unit = match.group(2)
    if unit == "M":
        return int(magnitude * 1e6)
    if unit == "B":
        return int(magnitude * 1e9)
    else:
        raise ValueError(f"Invalid model_size '{model_size}'. Expected forms like 50M, 2B, etc.")

def make_param_str(n_params: int) -> str:

    n_params = int(n_params)
    param_str = f"{n_params:_}"
    if n_params >= 1_000_000_000:
        param_str = f"{n_params // 1_000_000_000}B"
    elif n_params >= 1_000_000:
        param_str = f"{n_params // 1_000_000}M"

    return param_str

POWER_LAWS_PATH = (
    Path(__file__).resolve().parents[3] / "analysis" / "optimize" / "base" / "power_laws.json"
)

def load_power_law_params(path: Path) -> tuple[float, float, float, float]:
    data = json.loads(path.read_text())
    lr_coef = float(data["lr"]["coef"])
    lr_exp = float(data["lr"]["exp"])
    bs_coef = float(data["bs"]["coef"])
    bs_exp = float(data["bs"]["exp"])
    return lr_coef, lr_exp, bs_coef, bs_exp


_POWER_LAWS: tuple[float, float, float, float] | None = None


def _get_power_laws() -> tuple[float, float, float, float]:
    global _POWER_LAWS
    if _POWER_LAWS is None:
        _POWER_LAWS = load_power_law_params(POWER_LAWS_PATH)
    return _POWER_LAWS


def get_lr(n_params: int) -> float:
    lr_coef, lr_exp, _, _ = _get_power_laws()
    return lr_coef * (n_params ** lr_exp)


def get_bs(n_params: int) -> int:
    _, _, bs_coef, bs_exp = _get_power_laws()
    return max(1, round(bs_coef * (n_params ** bs_exp)))