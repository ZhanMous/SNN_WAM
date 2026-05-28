from __future__ import annotations

import csv
import json
from pathlib import Path

from src.train.eval_offline import run_eval_offline
from src.train.train_offline import run_training


ROOT = Path(__file__).resolve().parents[1]


def test_eval_offline_writes_eval_csv_and_reproducibility_files(tmp_path: Path) -> None:
    run_dir = run_training(
        ROOT / "configs/smoke/libero_spatial_gru_no_future.yaml",
        dry_run=True,
        max_steps=1,
        output_dir=tmp_path / "runs",
        run_id="eval_source",
        command=["python3", "src/train/train_offline.py", "--dry_run"],
    )

    output_csv = run_eval_offline(
        run_dir=run_dir,
        split="val",
        max_steps=1,
        device_name="cpu",
        command=["python3", "src/train/eval_offline.py", "--run_dir", str(run_dir)],
    )

    assert output_csv == run_dir / "eval_offline.csv"
    assert output_csv.exists()
    assert (run_dir / "eval_command.txt").exists()
    assert (run_dir / "eval_environment.json").exists()
    assert (run_dir / "eval_summary.json").exists()

    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["source_split"] == "val"
    assert rows[0]["split"] == "eval_val"
    assert float(rows[0]["action_mse"]) >= 0.0
    assert float(rows[0]["future_latent_cosine_error"]) >= 0.0
    assert len(json.loads(rows[0]["future_latent_cosine_error_by_horizon"])) == 4

    summary = json.loads((run_dir / "eval_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "offline_smoke_eval_not_closed_loop"
    assert "not_success_rate" in summary["non_claims"]
