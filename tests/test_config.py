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
    "configs/libero_spatial_wam_gru.yaml": "wam_gru",
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


def test_wam_placeholders_are_not_reportable_training_claims() -> None:
    for relative_path in [
        "configs/libero_spatial_wam_gru.yaml",
        "configs/libero_spatial_snn_lif.yaml",
    ]:
        config = load_config(ROOT / relative_path)

        assert config["data"]["future_horizon"] == 4
        assert "future_latent_contract_only" in config["experiment"]["tags"]
        assert config["training"]["lambda_future"] == 1.0


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
