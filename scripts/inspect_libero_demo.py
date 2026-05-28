#!/usr/bin/env python3
"""Locate and inspect the first real LIBERO demonstration file in a dataset root."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import inspect_libero_data  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTENSIONS = (".hdf5", ".h5", ".npz", ".json")

from _common import run_git  # noqa: E402


def configured_dataset_root(cli_root: Path | None) -> tuple[Path | None, str | None]:
    if cli_root is not None:
        return cli_root, "--dataset-root"
    for env_name in ("LIBERO_DATASET_ROOT", "LIBERO_DATA_ROOT"):
        value = os.environ.get(env_name)
        if value:
            return Path(value), env_name
    return None, None


def candidate_roots(dataset_root: Path, suite: str) -> list[Path]:
    return [
        dataset_root / suite,
        dataset_root / f"{suite}_no_noops",
        dataset_root / "datasets" / suite,
        dataset_root,
    ]


def find_demo_file(dataset_root: Path, suite: str) -> Path | None:
    seen: set[Path] = set()
    for root in candidate_roots(dataset_root, suite):
        if not root.exists() or root in seen:
            continue
        seen.add(root)
        for extension in DEFAULT_EXTENSIONS:
            matches = sorted(root.rglob(f"*{extension}"))
            if matches:
                return matches[0]
    return None


def missing_report(
    dataset_root: Path | None,
    root_source: str | None,
    suite: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "status": "missing",
        "reason": reason,
        "suite": suite,
        "dataset_root": str(dataset_root) if dataset_root else None,
        "dataset_root_source": root_source,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "git": {
            "commit": run_git(["rev-parse", "--short", "HEAD"]),
            "branch": run_git(["branch", "--show-current"]),
            "dirty": bool(run_git(["status", "--short"])),
        },
        "next_steps": [
            "Set LIBERO_DATASET_ROOT to a local LIBERO dataset directory.",
            "Install LIBERO and use its official dataset downloader for the minimal suite.",
            "Re-run scripts/inspect_libero_demo.py after a real demonstration file exists.",
        ],
    }


def write_missing_report(report: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{timestamp}_libero_demo_missing.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def update_docs(report: dict[str, Any], report_path: Path | None) -> None:
    contract = REPO_ROOT / "docs" / "LIBERO_DATA_CONTRACT.md"
    risks = REPO_ROOT / "docs" / "DATA_RISKS.md"

    if report.get("status") == "observed":
        write_observed_contract(contract, report, report_path)
        write_data_risks(risks, observed=True, report=report, report_path=report_path)
    else:
        write_unobserved_contract(contract, report)
        write_data_risks(risks, observed=False, report=report, report_path=report_path)


def write_unobserved_contract(path: Path, report: dict[str, Any]) -> None:
    path.write_text(
        f"""# LIBERO Data Contract

## Observed Schema

Status: not observed in this workspace.

No real LIBERO demonstration file has been inspected yet. Current blockers:

- LIBERO importable: no
- Dataset root configured: `{report.get('dataset_root')}`
- Dataset root source: `{report.get('dataset_root_source')}`
- Suite requested: `{report.get('suite')}`
- Missing reason: {report.get('reason')}

Do not proceed to `TrajectoryWindowDataset` until this section is replaced with a real file path and real observed schema.

## Required Real-Data Fields To Record

- Real demonstration file path.
- Trajectory group or archive key.
- Trajectory length `T`.
- Action shape and dtype, without assuming `action_dim`.
- Image keys, shapes, and dtypes.
- State/proprioception keys, shapes, and dtypes.
- Language/instruction keys and representation.
- Reward/success/done keys and confirmation they are labels only.
- Time indexing and action alignment convention.

## Official Download Note

Use the official LIBERO repository downloader from the installed LIBERO repo/README, normally `benchmark_scripts/download_libero_datasets.py --datasets libero_spatial` for the first minimal suite. Do not modify LIBERO source.
""",
        encoding="utf-8",
    )


def write_observed_contract(
    path: Path,
    report: dict[str, Any],
    report_path: Path | None,
) -> None:
    datasets = report.get("inspection", {}).get("datasets", [])
    image_records = [item for item in datasets if item.get("field_role") in {"image", "array_possible_image"}]
    state_records = [item for item in datasets if item.get("field_role") == "state"]
    action_records = [item for item in datasets if item.get("field_role") == "action"]
    language_records = [item for item in datasets if item.get("field_role") == "language"]
    time_info = report.get("inspection", {}).get("time_dimension_convention", {})

    def rows(records: list[dict[str, Any]]) -> str:
        if not records:
            return "| _none observed_ |  |  |  |\n"
        return "".join(
            f"| `{item['path']}` | `{item.get('shape')}` | `{item.get('dtype')}` | `{item.get('time_axis')}` |\n"
            for item in records
        )

    path.write_text(
        f"""# LIBERO Data Contract

## Observed Schema

- Status: observed
- Suite: `{report.get('suite')}`
- Dataset root: `{report.get('dataset_root')}`
- Demonstration file: `{report.get('demo_path')}`
- Inspection report: `{report_path}`
- Trajectory id: `{report.get('inspection', {}).get('trajectory_id')}`
- Time axis: `{time_info.get('axis')}`
- Observed time lengths: `{time_info.get('observed_time_lengths')}`
- Action alignment: {time_info.get('action_alignment')}
- Pre-observation gate: `not observed` / `blocked by G1.5` until a real LIBERO HDF5 demonstration was inspected.

### Actions

| Path | Shape | Dtype | Time Axis |
| --- | --- | --- | --- |
{rows(action_records)}
### Images

| Path | Shape | Dtype | Time Axis |
| --- | --- | --- | --- |
{rows(image_records)}
### State / Proprioception

| Path | Shape | Dtype | Time Axis |
| --- | --- | --- | --- |
{rows(state_records)}
### Language

| Path | Shape | Dtype | Time Axis |
| --- | --- | --- | --- |
{rows(language_records)}
## Time Indexing Convention

Use the inspected raw time axis as axis 0. For processed LIBERO HDF5, `docs/LIBERO_ACTION_SEMANTICS.md` defines the G2.5 convention: `actions[t]` led to `obs[t]`, so policy targets after `image_t = obs[t]` start at `actions[t+1]`.

## Future Leakage Rule

Inputs may include observation/state at `t`, instruction, and already executed actions through `actions[t]` under the processed-HDF5 convention. Targets after `image_t = obs[t]` start at `actions[t+1]`; future images/latents also start at `t+1`.
""",
        encoding="utf-8",
    )


def write_data_risks(
    path: Path,
    observed: bool,
    report: dict[str, Any],
    report_path: Path | None,
) -> None:
    datasets = report.get("inspection", {}).get("datasets", [])
    roles = {item.get("field_role") for item in datasets}
    time_info = report.get("inspection", {}).get("time_dimension_convention", {})
    action_alignment = str(time_info.get("action_alignment", ""))

    demo_status = "resolved" if observed and report.get("demo_path") else "unresolved"
    action_status = "resolved" if observed and "action" in roles else "blocked by G1.5"
    camera_status = (
        "resolved"
        if observed and ("image" in roles or "array_possible_image" in roles)
        else "blocked by G1.5"
    )
    state_status = "resolved" if observed and "state" in roles else "blocked by G1.5"
    language_status = "resolved" if observed and "language" in roles else "unresolved"
    split_status = "unresolved" if observed else "blocked by G1.5"
    alignment_status = (
        "unresolved"
        if observed and "unverified" in action_alignment.lower()
        else ("resolved" if observed else "blocked by G1.5")
    )

    path.write_text(
        f"""# Data Risks

## Current Status

- Real LIBERO demonstration inspected: `{observed}`
- Dataset root: `{report.get('dataset_root')}`
- Demonstration path: `{report.get('demo_path')}`
- Inspection report: `{report_path}`
- Previous G1.5 state before real-data inspection: `not observed` / `blocked by G1.5`.

## Risk Register

| Risk | Status | Notes |
| --- | --- | --- |
| Real demonstration file path unknown | {demo_status} | Must be recorded before dataset implementation. |
| Real action dimension unknown | {action_status} | Do not assume action dimension before one real HDF5 demonstration is inspected. |
| Camera keys unknown | {camera_status} | Must inspect real image/camera keys before choosing inputs. |
| State/proprio keys unknown | {state_status} | Must identify real state/proprio fields before deciding whether state is input or audit-only. |
| Language keys unknown | {language_status} | Must identify real instruction field source. |
| Split format unknown | {split_status} | Must inspect official file layout and split metadata before defining train/val/test policy. |
| Action alignment unknown | {alignment_status} | Use `docs/LIBERO_ACTION_SEMANTICS.md`; fresh replay validation remains future work. |
| Future leakage through inputs | partially resolved | Synthetic trajectory-window tests cover action, image, and dry-run future-latent alignment; real frozen latent extraction remains future work. |
| Split leakage | unresolved | Requires train/val/test split policy and no normalization on val/test. |

## Do Not Proceed

Do not proceed to real-data WAM-style future-latent claims until frozen visual latents are precomputed or adapter-produced with recorded metadata. Offline dry-run WAM training may run only as a smoke test and must follow `docs/LIBERO_ACTION_SEMANTICS.md`, `docs/SPLIT_POLICY.md`, and `docs/NORMALIZATION_POLICY.md`.
""",
        encoding="utf-8",
    )


def observed_report(
    dataset_root: Path,
    root_source: str,
    suite: str,
    demo_path: Path,
    inspection: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "observed",
        "suite": suite,
        "dataset_root": str(dataset_root),
        "dataset_root_source": root_source,
        "demo_path": str(demo_path),
        "inspection": inspection,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--output-dir", type=Path, default=Path("results/inspections"))
    parser.add_argument("--trajectory", help="Optional HDF5 trajectory group.")
    parser.add_argument("--update-docs", action="store_true")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Exit 0 and write a missing report when no real demo can be found.",
    )
    args = parser.parse_args()

    dataset_root, root_source = configured_dataset_root(args.dataset_root)
    if dataset_root is None:
        report = missing_report(None, None, args.suite, "dataset_root_not_configured")
        path = write_missing_report(report, args.output_dir)
        if args.update_docs:
            update_docs(report, path)
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"inspection_report={path}")
        return 0 if args.allow_missing else 1

    dataset_root = dataset_root.expanduser().resolve()
    if not dataset_root.exists():
        report = missing_report(dataset_root, root_source, args.suite, "dataset_root_missing")
        path = write_missing_report(report, args.output_dir)
        if args.update_docs:
            update_docs(report, path)
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"inspection_report={path}")
        return 0 if args.allow_missing else 1

    demo_path = find_demo_file(dataset_root, args.suite)
    if demo_path is None:
        report = missing_report(dataset_root, root_source, args.suite, "no_demo_file_found")
        path = write_missing_report(report, args.output_dir)
        if args.update_docs:
            update_docs(report, path)
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"inspection_report={path}")
        return 0 if args.allow_missing else 1

    inspection = inspect_libero_data.inspect_real(demo_path, args.trajectory)
    inspection_path = inspect_libero_data.write_report(inspection, args.output_dir, "json")
    report = observed_report(dataset_root, root_source or "unknown", args.suite, demo_path, inspection)
    if args.update_docs:
        update_docs(report, inspection_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"inspection_report={inspection_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
