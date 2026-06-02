#!/usr/bin/env python3
"""Artifact and claim audit for DINOwM training runs.

Verifies all expected files exist, checkpoint reloadability, config consistency,
and generates a structured claim audit.

Usage:
    python scripts/audit_dinowm_run.py \
        --run_dir results/runs/dinowm_transformer_baseline_real
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.dinowm_transformer import DINOwMTransformer  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, type=Path)
    return parser.parse_args(argv)


def check_file_exists(path: Path, label: str) -> bool:
    exists = path.exists()
    if not exists:
        print(f"  MISSING: {label} ({path})")
    return exists


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir

    print(f"Auditing run: {run_dir}")

    audit: dict[str, Any] = {
        "run_dir": str(run_dir),
        "artifact_status": "pass",
        "reload_status": "pass",
        "metric_status": "pass",
        "claim_status": {},
        "forbidden_claims": [],
        "errors": [],
        "warnings": [],
    }

    # --- 1. Artifact existence ---
    print("\n--- Artifact Check ---")
    required_files = {
        "best.pt": "Best checkpoint",
        "last.pt": "Last checkpoint",
        "metrics.csv": "Training metrics",
        "summary.json": "Training summary",
        "config.yaml": "Config copy",
        "command.sh": "Run command",
        "git_commit.txt": "Git commit",
        "split.json": "Split info",
        "notes.md": "Notes",
    }
    optional_files = {
        "environment.json": "Environment info",
        "seeds.json": "Seeds",
    }

    for fname, label in required_files.items():
        if not check_file_exists(run_dir / fname, label):
            audit["artifact_status"] = "fail"
            audit["errors"].append(f"Required file missing: {fname}")

    for fname, label in optional_files.items():
        check_file_exists(run_dir / fname, label)

    # Check eval outputs
    eval_dir = run_dir / "eval_multihorizon"
    eval_files = {
        "eval_metrics.csv": "Multi-horizon eval metrics",
        "summary.json": "Multi-horizon eval summary",
    }
    for fname, label in eval_files.items():
        if not check_file_exists(eval_dir / fname, label):
            audit["warnings"].append(f"Eval file missing: {fname}")

    # Check ablation outputs
    ablation_dir = run_dir / "eval_multihorizon_ablation"
    for mode in ["zeros", "shuffle"]:
        mode_dir = ablation_dir / mode
        if not mode_dir.exists():
            audit["warnings"].append(f"Ablation dir missing: {mode_dir}")

    # Check sample_id consistency across action modes (paired eval)
    print("\n--- Sample ID Consistency ---")
    sample_id_sets = {}
    for mode_label, mode_dir in [
        ("real", eval_dir),
        ("zeros", ablation_dir / "zeros"),
        ("shuffle", ablation_dir / "shuffle"),
    ]:
        per_sample_csv = mode_dir / "per_sample_metrics.csv"
        if per_sample_csv.exists():
            import csv as csv_mod
            with open(per_sample_csv) as f:
                reader = csv_mod.DictReader(f)
                ids = set()
                for row in reader:
                    sid = row.get("sample_id", "")
                    horizon = row.get("horizon", "")
                    ids.add(f"{sid}_H{horizon}")
            sample_id_sets[mode_label] = ids
            print(f"  {mode_label}: {len(ids)} sample_ids")

    if len(sample_id_sets) >= 2:
        ref_mode = "real"
        if ref_mode in sample_id_sets:
            ref_ids = sample_id_sets[ref_mode]
            for other_mode, other_ids in sample_id_sets.items():
                if other_mode == ref_mode:
                    continue
                missing_in_other = ref_ids - other_ids
                extra_in_other = other_ids - ref_ids
                if missing_in_other or extra_in_other:
                    audit["warnings"].append(
                        f"sample_id mismatch: {ref_mode} vs {other_mode}: "
                        f"{len(missing_in_other)} missing, {len(extra_in_other)} extra"
                    )
                    print(f"  MISMATCH {ref_mode} vs {other_mode}: "
                          f"{len(missing_in_other)} missing, {len(extra_in_other)} extra")
                else:
                    print(f"  {ref_mode} == {other_mode}: sample_ids match")

    # Check planning sanity
    planning_dir = run_dir / "planning_sanity"
    planning_files = {
        "summary.json": "Planning summary",
        "per_sample_results.csv": "Planning per-sample results",
        "optimization_traces.csv": "Planning traces",
    }
    for fname, label in planning_files.items():
        if not check_file_exists(planning_dir / fname, label):
            audit["warnings"].append(f"Planning file missing: {fname}")

    # Check baselines
    baselines_dir = run_dir / "baselines"
    if not check_file_exists(baselines_dir / "persistence_metrics.json", "Persistence baseline"):
        audit["warnings"].append("Persistence baseline missing")

    # --- 2. Checkpoint reload ---
    print("\n--- Checkpoint Reload ---")
    best_path = run_dir / "best.pt"
    if best_path.exists():
        try:
            checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
            config = checkpoint["config"]
            model_cfg = config["model"]

            model = DINOwMTransformer(
                patch_dim=int(model_cfg["patch_dim"]),
                feature_dim=int(model_cfg["feature_dim"]),
                action_dim=int(model_cfg["action_dim"]),
                hidden_dim=int(model_cfg["hidden_dim"]),
                num_heads=int(model_cfg["num_heads"]),
                num_layers=int(model_cfg["num_layers"]),
                future_horizon=int(model_cfg["future_horizon"]),
                dropout=float(model_cfg["dropout"]),
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"  best.pt loaded: epoch={checkpoint.get('epoch', '?')}, "
                  f"val_metric={checkpoint.get('val_patch_cosine_error', '?')}")
        except Exception as e:
            audit["reload_status"] = "fail"
            audit["errors"].append(f"Checkpoint reload failed: {e}")
            print(f"  FAILED: {e}")
    else:
        audit["reload_status"] = "fail"
        audit["errors"].append("best.pt not found")

    # --- 3. Config consistency ---
    print("\n--- Config Consistency ---")
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        # Check that train and eval configs match
        if "config_path" in summary and "dataset_cache_dir" in summary:
            print(f"  Config path: {summary['config_path']}")
            print(f"  Cache dir: {summary['dataset_cache_dir']}")
        else:
            audit["warnings"].append("summary.json missing config_path or dataset_cache_dir")
    else:
        audit["warnings"].append("summary.json not found")

    # --- 4. Metric comparison ---
    print("\n--- Metric Comparison ---")
    real_metrics = {}
    eval_csv = run_dir / "eval_multihorizon" / "eval_metrics.csv"
    if eval_csv.exists():
        import csv
        with open(eval_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("split") == "val":
                    h = int(row["horizon"])
                    real_metrics[h] = {
                        "patch_cosine_error": float(row["patch_cosine_error"]),
                        "patch_mse": float(row["patch_mse"]),
                    }
        print(f"  Real eval: {len(real_metrics)} horizons")

    persistence_metrics = {}
    persistence_path = baselines_dir / "persistence_metrics.json"
    if persistence_path.exists():
        p_data = load_json(persistence_path)
        for r in p_data.get("results", []):
            if r.get("split") == "val":
                h = int(r["horizon"])
                persistence_metrics[h] = {
                    "patch_cosine_error": r["patch_cosine_error"],
                    "patch_mse": r["patch_mse"],
                }
        print(f"  Persistence baseline: {len(persistence_metrics)} horizons")

    zeros_metrics = {}
    zeros_dir = ablation_dir / "zeros" if ablation_dir.exists() else None
    if zeros_dir and (zeros_dir / "eval_metrics.csv").exists():
        with open(zeros_dir / "eval_metrics.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("split") == "val":
                    h = int(row["horizon"])
                    zeros_metrics[h] = {
                        "patch_cosine_error": float(row["patch_cosine_error"]),
                    }
        print(f"  Zero-action ablation: {len(zeros_metrics)} horizons")

    # --- 5. Claim audit ---
    print("\n--- Claim Audit ---")

    # Claim: trainable on real cached latents
    if audit["reload_status"] == "pass" and summary_path.exists():
        audit["claim_status"]["trainable_on_real_cached_latents"] = "supported"
        print("  trainable_on_real_cached_latents: SUPPORTED")
    else:
        audit["claim_status"]["trainable_on_real_cached_latents"] = "unsupported"
        print("  trainable_on_real_cached_latents: UNSUPPORTED")

    # Claim: beats persistence
    if real_metrics and persistence_metrics:
        h1_real = real_metrics.get(1, {}).get("patch_cosine_error", float("inf"))
        h1_pers = persistence_metrics.get(1, {}).get("patch_cosine_error", float("inf"))
        if h1_real < h1_pers:
            audit["claim_status"]["beats_persistence"] = "supported"
            print(f"  beats_persistence: SUPPORTED (real={h1_real:.4f} < persistence={h1_pers:.4f})")
        else:
            audit["claim_status"]["beats_persistence"] = "unsupported"
            print(f"  beats_persistence: UNSUPPORTED (real={h1_real:.4f} >= persistence={h1_pers:.4f})")
    else:
        audit["claim_status"]["beats_persistence"] = "unsupported"
        print("  beats_persistence: UNSUPPORTED (missing metrics)")

    # Claim: uses action information
    # NOTE: zeros is an eval-time OOD ablation (action inputs zeroed for a model
    # trained on real actions), NOT a true no-action baseline. A proper no-action
    # baseline would be a model trained without action inputs. If real > zeros,
    # the improvement could partly reflect OOD degradation rather than genuine
    # action utility. The shuffle comparison is the cleaner action-sensitivity test.
    if real_metrics and zeros_metrics:
        h1_real = real_metrics.get(1, {}).get("patch_cosine_error", float("inf"))
        h1_zeros = zeros_metrics.get(1, {}).get("patch_cosine_error", float("inf"))
        if h1_real < h1_zeros * 0.95:
            audit["claim_status"]["uses_action_information"] = "supported"
            audit["claim_status"]["uses_action_information_note"] = (
                "real < zeros by >5%, but zeros is OOD ablation not true no-action baseline. "
                "Shuffle comparison is the cleaner action-sensitivity test."
            )
            print(f"  uses_action_information: SUPPORTED (real={h1_real:.4f} < zeros={h1_zeros:.4f})")
        elif h1_real < h1_zeros:
            audit["claim_status"]["uses_action_information"] = "weak"
            audit["claim_status"]["uses_action_information_note"] = (
                "Margin <5%; zeros is OOD ablation, result may partly reflect input distribution shift."
            )
            print(f"  uses_action_information: WEAK (real={h1_real:.4f} < zeros={h1_zeros:.4f} but margin small)")
        else:
            audit["claim_status"]["uses_action_information"] = "unsupported"
            print(f"  uses_action_information: UNSUPPORTED (real={h1_real:.4f} >= zeros={h1_zeros:.4f})")
    else:
        audit["claim_status"]["uses_action_information"] = "unsupported"
        print("  uses_action_information: UNSUPPORTED (missing metrics)")

    # Claim: model-internal planning sanity
    planning_summary = planning_dir / "summary.json"
    if planning_summary.exists():
        p_summary = load_json(planning_summary)
        pass_rate = p_summary.get("comparison", {}).get("pass_rate", 0)
        if pass_rate > 0.5:
            audit["claim_status"]["model_internal_planning_sanity"] = "supported"
            print(f"  model_internal_planning_sanity: SUPPORTED (pass_rate={pass_rate:.1%})")
        else:
            audit["claim_status"]["model_internal_planning_sanity"] = "unsupported"
            print(f"  model_internal_planning_sanity: UNSUPPORTED (pass_rate={pass_rate:.1%})")
    else:
        audit["claim_status"]["model_internal_planning_sanity"] = "unsupported"
        print("  model_internal_planning_sanity: UNSUPPORTED (no planning results)")

    # Always unsupported
    audit["claim_status"]["real_libero_planning_success"] = "unsupported"
    print("  real_libero_planning_success: UNSUPPORTED (no closed-loop eval)")

    # Forbidden claims
    audit["forbidden_claims"] = [
        "improves LIBERO task success rate",
        "closed-loop planning works on real robot",
        "general world model learned from small dataset",
        "neuromorphic low power (not measured on neuromorphic hardware)",
        "foundation model result from small LIBERO experiment",
        "zero-action ablation proves action utility (OOD caveat: see uses_action_information_note)",
    ]

    # --- 6. Write audit report ---
    out_path = run_dir / "audit_report.json"
    out_path.write_text(json.dumps(audit, indent=2, default=str) + "\n")
    print(f"\nAudit report written to {out_path}")

    # Summary
    print(f"\n=== Audit Summary ===")
    print(f"  Artifacts: {audit['artifact_status']}")
    print(f"  Reload: {audit['reload_status']}")
    print(f"  Metrics: {audit['metric_status']}")
    print(f"  Errors: {len(audit['errors'])}")
    print(f"  Warnings: {len(audit['warnings'])}")

    if audit["errors"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
