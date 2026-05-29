"""Config loading and validation for experiment placeholders.

The schema is intentionally small and explicit. It validates infrastructure
contracts only; it does not instantiate datasets, models, or training loops.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigValidationError(ValueError):
    """Raised when a config is missing required experiment fields."""


REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "experiment": ("name", "seed", "tags"),
    "data": (
        "suite",
        "dataset_root",
        "history_len",
        "action_horizon",
        "future_horizon",
        "image_size",
        "split",
    ),
    "model": ("visual_encoder", "text_encoder", "temporal_adapter", "hidden_dim"),
    "training": (
        "batch_size",
        "epochs",
        "optimizer",
        "lr",
        "lambda_action",
        "lambda_future",
        "lambda_spike",
        "grad_clip_norm",
    ),
    "output": ("output_dir", "save_best_by"),
}

# Optional fields that, if present, must satisfy their own validation rules.
OPTIONAL_SECTIONS: dict[str, tuple[str, ...]] = {
    "reproducibility": ("require_clean_git",),
}

ALLOWED_TEMPORAL_ADAPTERS = {"mlp", "gru", "wam_gru", "bc_gru", "snn_lif"}
ALLOWED_OPTIMIZERS = {"adamw"}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML experiment config.

    Args:
        path: YAML file path.

    Returns:
        A plain dictionary preserving the YAML structure.

    Raises:
        ConfigValidationError: If required sections or fields are missing, or
            if scalar values violate the current placeholder schema.
    """

    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigValidationError(f"{config_path} must contain a YAML mapping")
    validate_config(data)
    return data


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate the common Phase-1 experiment config shape."""

    if not isinstance(config, Mapping):
        raise ConfigValidationError("config must be a mapping")

    for section, fields in REQUIRED_FIELDS.items():
        value = config.get(section)
        if not isinstance(value, Mapping):
            raise ConfigValidationError(f"missing required section: {section}")
        missing = [field for field in fields if field not in value]
        if missing:
            raise ConfigValidationError(f"section {section} missing fields: {missing}")

    # Validate optional sections if present.
    for section, fields in OPTIONAL_SECTIONS.items():
        value = config.get(section)
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise ConfigValidationError(f"optional section {section} must be a mapping")
        missing = [field for field in fields if field not in value]
        if missing:
            raise ConfigValidationError(f"section {section} missing fields: {missing}")

    experiment = config["experiment"]
    data = config["data"]
    model = config["model"]
    training = config["training"]
    output = config["output"]

    _require_nonempty_string(experiment["name"], "experiment.name")
    _require_int(experiment["seed"], "experiment.seed", minimum=0)
    if not isinstance(experiment["tags"], list):
        raise ConfigValidationError("experiment.tags must be a list")

    _require_nonempty_string(data["suite"], "data.suite")
    _validate_dataset_root(data["dataset_root"])
    _require_int(data["history_len"], "data.history_len", minimum=1)
    _require_int(data["action_horizon"], "data.action_horizon", minimum=1)
    _require_int(data["future_horizon"], "data.future_horizon", minimum=0)
    _require_int(data["image_size"], "data.image_size", minimum=1)
    _validate_split(data["split"])

    _require_nonempty_string(model["visual_encoder"], "model.visual_encoder")
    _require_nonempty_string(model["text_encoder"], "model.text_encoder")
    if model["temporal_adapter"] not in ALLOWED_TEMPORAL_ADAPTERS:
        raise ConfigValidationError(
            "model.temporal_adapter must be one of "
            f"{sorted(ALLOWED_TEMPORAL_ADAPTERS)}"
        )
    _require_int(model["hidden_dim"], "model.hidden_dim", minimum=1)

    _require_int(training["batch_size"], "training.batch_size", minimum=1)
    _require_int(training["epochs"], "training.epochs", minimum=0)
    if training["optimizer"] not in ALLOWED_OPTIMIZERS:
        raise ConfigValidationError(
            f"training.optimizer must be one of {sorted(ALLOWED_OPTIMIZERS)}"
        )
    _require_float(training["lr"], "training.lr", minimum=0.0, strictly_positive=True)
    for field in ("lambda_action", "lambda_future", "lambda_spike"):
        _require_float(training[field], f"training.{field}", minimum=0.0)
    grad_clip = training["grad_clip_norm"]
    if grad_clip is not None:
        _require_float(grad_clip, "training.grad_clip_norm", minimum=0.0)

    _require_nonempty_string(output["output_dir"], "output.output_dir")
    _require_nonempty_string(output["save_best_by"], "output.save_best_by")

    # Validate reproducibility section if present.
    reproducibility = config.get("reproducibility")
    if reproducibility is not None:
        if not isinstance(reproducibility.get("require_clean_git"), bool):
            raise ConfigValidationError(
                "reproducibility.require_clean_git must be a boolean"
            )


def _validate_dataset_root(value: Any) -> None:
    _require_nonempty_string(value, "data.dataset_root")
    root = str(value)
    if root.startswith(("env:", "$", "${")):
        return
    path = Path(root).expanduser()
    if path.is_absolute() or root.startswith("~"):
        raise ConfigValidationError(
            "data.dataset_root must be an environment reference or relative path; "
            "do not hard-code local absolute dataset paths in committed configs"
        )
    if ".." in Path(root).parts:
        raise ConfigValidationError(
            "data.dataset_root must not contain path traversal sequences (..)"
        )


def _validate_split(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ConfigValidationError("data.split must be a mapping")
    for field in ("source", "unit", "train", "val", "test"):
        if field not in value:
            raise ConfigValidationError(f"data.split missing field: {field}")
    _require_nonempty_string(value["source"], "data.split.source")
    _require_nonempty_string(value["unit"], "data.split.unit")
    for field in ("train", "val", "test"):
        if not isinstance(value[field], list):
            raise ConfigValidationError(f"data.split.{field} must be a list")


def _require_nonempty_string(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ConfigValidationError(f"{name} must be a non-empty string")


def _require_int(value: Any, name: str, *, minimum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ConfigValidationError(f"{name} must be an integer >= {minimum}")


def _require_float(
    value: Any,
    name: str,
    *,
    minimum: float,
    strictly_positive: bool = False,
) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigValidationError(f"{name} must be numeric")
    if strictly_positive and float(value) <= minimum:
        raise ConfigValidationError(f"{name} must be > {minimum}")
    if not strictly_positive and float(value) < minimum:
        raise ConfigValidationError(f"{name} must be >= {minimum}")


__all__ = [
    "ALLOWED_TEMPORAL_ADAPTERS",
    "ConfigValidationError",
    "REQUIRED_FIELDS",
    "load_config",
    "validate_config",
]
