from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.utils.config import (
    ALLOWED_TEMPORAL_ADAPTERS,
    ConfigValidationError,
    load_config,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]


CONFIGS = {
    "configs/libero_spatial_mlp.yaml": "mlp",
    "configs/libero_spatial_gru.yaml": "gru",
    "configs/smoke/libero_spatial_wam_gru.yaml": "wam_gru",
    "configs/smoke/libero_spatial_gru_no_future.yaml": "wam_gru",
    "configs/libero_spatial_snn_lif.yaml": "snn_lif",
}


def test_placeholder_configs_load_and_validate() -> None:
    for relative_path, adapter in CONFIGS.items():
        config = load_config(ROOT / relative_path)

        assert config["data"]["suite"] == "libero_spatial"
        assert config["data"]["dataset_root"] == "env:LIBERO_DATASET_ROOT"
        assert config["model"]["temporal_adapter"] == adapter
        assert adapter in ALLOWED_TEMPORAL_ADAPTERS
        assert config["training"]["epochs"] == 0


def test_action_only_smoke_config_loads_and_stays_in_mlp_scope() -> None:
    config = load_config(ROOT / "configs/smoke/libero_spatial_action_only_smoke.yaml")

    assert config["data"]["suite"] == "libero_spatial"
    assert config["data"]["dataset_root"] == "env:LIBERO_DATASET_ROOT"
    assert config["data"]["future_horizon"] == 0
    assert config["data"]["max_train_trajectories"] == 2
    assert config["data"]["max_val_trajectories"] == 1
    assert config["model"]["model_name"] == "action_only_mlp_smoke"
    assert config["model"]["visual_encoder"] == "stub"
    assert config["model"]["text_encoder"] == "stub"
    assert config["model"]["temporal_adapter"] == "mlp"
    assert config["normalization"]["actions"]["mode"] == "standardize_train"
    assert config["normalization"]["actions"]["fit_split"] == "train"
    assert config["training"]["epochs"] == 3
    assert config["training"]["lambda_future"] == 0.0
    assert config["training"]["lambda_spike"] == 0.0


def test_wam_placeholders_are_not_reportable_training_claims() -> None:
    wam_config = load_config(ROOT / "configs/smoke/libero_spatial_wam_gru.yaml")

    assert wam_config["data"]["future_horizon"] == 4
    assert "g4_ablation" in wam_config["experiment"]["tags"]
    assert "future_latent_smoke" in wam_config["experiment"]["tags"]
    assert wam_config["model"]["visual_encoder"] == "smoke_time_index"
    assert wam_config["model"]["visual_latent_dim"] == 8
    assert wam_config["training"]["lambda_future"] == 1.0

    snn_config = load_config(ROOT / "configs/libero_spatial_snn_lif.yaml")

    assert snn_config["data"]["future_horizon"] == 4
    assert "future_latent_contract_only" in snn_config["experiment"]["tags"]
    assert snn_config["training"]["lambda_future"] == 1.0


def test_wam_gru_ablation_configs_differ_only_in_future_objective_metadata() -> None:
    with_future = load_config(ROOT / "configs/smoke/libero_spatial_wam_gru.yaml")
    no_future = load_config(ROOT / "configs/smoke/libero_spatial_gru_no_future.yaml")

    allowed_differences = {
        ("experiment", "name"),
        ("experiment", "tags"),
        ("training", "lambda_future"),
    }
    differences = {
        path
        for path, left, right in compare_mappings(with_future, no_future)
        if left != right
    }

    assert differences == allowed_differences
    assert with_future["training"]["lambda_future"] == 1.0
    assert no_future["training"]["lambda_future"] == 0.0
    assert with_future["model"] == no_future["model"]
    assert with_future["data"] == no_future["data"]


def test_config_validation_rejects_missing_required_section() -> None:
    config = load_config(ROOT / "configs/libero_spatial_mlp.yaml")
    bad = deepcopy(config)
    bad.pop("model")

    with pytest.raises(ConfigValidationError, match="missing required section: model"):
        validate_config(bad)


def test_config_validation_rejects_absolute_dataset_root(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/libero_spatial_mlp.yaml")
    bad = deepcopy(config)
    bad["data"]["dataset_root"] = "/home/user/local/libero"

    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(bad), encoding="utf-8")

    with pytest.raises(ConfigValidationError, match="do not hard-code"):
        load_config(path)


def compare_mappings(
    left: dict[str, object],
    right: dict[str, object],
    prefix: tuple[str, ...] = (),
) -> list[tuple[tuple[str, ...], object, object]]:
    keys = sorted(set(left) | set(right))
    differences: list[tuple[tuple[str, ...], object, object]] = []
    for key in keys:
        left_value = left.get(key)
        right_value = right.get(key)
        path = (*prefix, key)
        if isinstance(left_value, dict) and isinstance(right_value, dict):
            differences.extend(compare_mappings(left_value, right_value, path))
        else:
            differences.append((path, left_value, right_value))
    return differences
