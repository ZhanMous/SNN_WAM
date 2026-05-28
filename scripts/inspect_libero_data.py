#!/usr/bin/env python3
"""Inspect one LIBERO demonstration trajectory and write a raw data report."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def shape_list(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]


def dtype_name(value: Any) -> str | None:
    dtype = getattr(value, "dtype", None)
    return str(dtype) if dtype is not None else None


def dataset_record(
    path: str,
    shape: list[int] | None,
    dtype: str | None,
    *,
    source_type: str = "dataset",
    value_preview: str | None = None,
) -> dict[str, Any]:
    record = {
        "path": path,
        "shape": shape,
        "dtype": dtype,
        "time_axis": infer_time_axis(shape),
        "field_role": infer_field_role(path, shape),
        "source_type": source_type,
    }
    if value_preview is not None:
        record["value_preview"] = value_preview
    return record


def infer_time_axis(shape: list[int] | None) -> int | None:
    if not shape:
        return None
    if len(shape) >= 1 and shape[0] > 1:
        return 0
    return None


def infer_field_role(path: str, shape: list[int] | None) -> str:
    key = path.lower()
    leaf = key.rsplit("/", maxsplit=1)[-1]
    if "language" in key or "instruction" in key:
        return "language"
    if leaf in {"actions", "action"} or key.endswith("/actions"):
        return "action"
    if "reward" in key:
        return "reward"
    if "success" in key or "done" in key or "terminal" in key:
        return "success_or_done"
    if any(token in key for token in ["rgb", "image", "agentview", "eye_in_hand"]):
        return "image"
    if any(
        token in key
        for token in ["state", "proprio", "eef", "ee_pos", "ee_ori", "joint", "robot"]
    ):
        return "state"
    if shape and len(shape) >= 3:
        return "array_possible_image"
    if shape and len(shape) == 2:
        return "array_possible_timeseries"
    return "unknown"


def command_record() -> dict[str, Any]:
    status_short = run_git(["status", "--short"])
    return {
        "command": sys.argv,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "git": {
            "commit": run_git(["rev-parse", "--short", "HEAD"]),
            "branch": run_git(["branch", "--show-current"]),
            "dirty": bool(status_short),
        },
    }


def mock_report() -> dict[str, Any]:
    datasets = [
        dataset_record("trajectory/obs/agentview_rgb", [8, 128, 128, 3], "uint8"),
        dataset_record("trajectory/obs/robot_state", [8, 10], "float32"),
        dataset_record("trajectory/actions", [8, 7], "float32"),
        dataset_record("trajectory/rewards", [8], "float32"),
        dataset_record("trajectory/dones", [8], "bool"),
        dataset_record("trajectory/language", None, "str"),
    ]
    return {
        **command_record(),
        "mode": "mock",
        "mock": True,
        "source": "synthetic mock trajectory; no real LIBERO data was inspected",
        "trajectory_id": "mock_trajectory_0",
        "trajectory_keys": [record["path"] for record in datasets],
        "datasets": datasets,
        "language": "mock instruction: put the object in the bowl",
        "time_dimension_convention": {
            "axis": 0,
            "observed_time_lengths": [8],
            "meaning": "time T is the first dimension for image/state/action arrays in the mock report",
            "action_alignment": "mock follows documented action_to_current_obs convention unless overridden by tests",
        },
        "dataset_item_contract_preview": dataset_item_contract_preview(),
        "future_leakage_risks": future_leakage_risks(),
        "unresolved_questions": unresolved_questions(real_data=False),
    }


def dataset_item_contract_preview() -> dict[str, str]:
    return {
        "image_t": "[H, W, C] raw image before any final dataset transform",
        "instruction": "string",
        "action_history": "[history_len, action_dim], actions before the current policy decision",
        "state_t": "[state_dim] if present and used",
        "target_actions": "[action_horizon, action_dim], future actions per documented action semantics",
        "target_future_images": "[future_horizon, H, W, C], images from t+1 onward",
    }


def future_leakage_risks() -> list[str]:
    return [
        "action_history must not include future target actions",
        "future images or future states must never appear in model inputs",
        "reward, success, done, and episode outcome fields are evaluation labels, not inputs",
        "normalization statistics must be fit on train split only",
        "language/task metadata must not encode held-out split labels or success outcomes",
    ]


def unresolved_questions(real_data: bool) -> list[str]:
    base = [
        "Optional stronger validation: replay one processed trajectory and compare returned observations frame-by-frame against stored HDF5 images.",
        "Confirm exact camera keys to use as primary image inputs.",
        "Confirm whether robot state is required for Phase-1 baselines or only audited.",
        "Implement the real split-aware loader that materializes docs/SPLIT_POLICY.md.",
        "Confirm how initial states are stored for closed-loop evaluation.",
    ]
    if not real_data:
        base.insert(0, "Real LIBERO demonstration file was unavailable; all observed shapes are mock-only.")
    return base


def find_hdf5_trajectory_group(handle: Any, requested: str | None) -> tuple[str, Any]:
    if requested:
        return requested, handle[requested]
    if "data" in handle:
        data_group = handle["data"]
        for key in sorted(data_group.keys()):
            if hasattr(data_group[key], "keys"):
                return f"data/{key}", data_group[key]
    for candidate in ["demo_0", "trajectory", "traj_0"]:
        if candidate in handle:
            return candidate, handle[candidate]
    return "/", handle


def hdf5_attrs_by_path(handle: Any, trajectory_id: str) -> dict[str, dict[str, Any]]:
    paths = ["/"]
    if trajectory_id != "/":
        current = ""
        for part in trajectory_id.strip("/").split("/"):
            current = f"{current}/{part}" if current else part
            paths.append(current)

    attrs: dict[str, dict[str, Any]] = {}
    for path in paths:
        obj = handle if path == "/" else handle[path]
        values = {key: stringify_attr(value) for key, value in obj.attrs.items()}
        if values:
            attrs[path] = values
    return attrs


def try_parse_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def preview_value(value: Any, limit: int = 160) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, sort_keys=True)
    else:
        text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def language_records_from_attrs(attrs_by_path: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def visit(path: str, value: Any) -> None:
        parsed = try_parse_json(value)
        if parsed is not value:
            visit(path, parsed)
            return
        if isinstance(value, dict):
            for key, item in value.items():
                visit(f"{path}/{key}", item)
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                visit(f"{path}/{index}", item)
            return
        leaf = path.rsplit("/", maxsplit=1)[-1].lower()
        if "language" not in leaf and "instruction" not in leaf:
            return
        preview = preview_value(value)
        identity = (path, preview)
        if identity in seen:
            return
        seen.add(identity)
        records.append(
            dataset_record(
                path,
                None,
                type(value).__name__,
                source_type="attribute",
                value_preview=preview,
            )
        )

    for attr_path, attrs in attrs_by_path.items():
        normalized = attr_path.strip("/") or "root"
        for key, value in attrs.items():
            visit(f"attrs/{normalized}/{key}", value)
    return records


def inspect_hdf5(path: Path, trajectory: str | None) -> dict[str, Any]:
    try:
        import h5py  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise SystemExit("h5py is required to inspect HDF5 LIBERO files.") from exc

    datasets: list[dict[str, Any]] = []
    attrs: dict[str, Any] = {}
    attrs_by_path: dict[str, dict[str, Any]] = {}
    with h5py.File(path, "r") as handle:
        trajectory_id, group = find_hdf5_trajectory_group(handle, trajectory)
        attrs_by_path = hdf5_attrs_by_path(handle, trajectory_id)
        attrs = attrs_by_path.get(trajectory_id, {})

        def visitor(name: str, obj: Any) -> None:
            if hasattr(obj, "shape") and hasattr(obj, "dtype"):
                datasets.append(dataset_record(name, shape_list(obj), dtype_name(obj)))

        group.visititems(visitor)
        datasets.extend(language_records_from_attrs(attrs_by_path))

    return real_report(path, "hdf5", trajectory_id, datasets, attrs, attrs_by_path)


def inspect_npz(path: Path) -> dict[str, Any]:
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise SystemExit("NumPy is required to inspect NPZ files.") from exc

    datasets: list[dict[str, Any]] = []
    with np.load(path, allow_pickle=False) as archive:
        for key in sorted(archive.files):
            array = archive[key]
            datasets.append(dataset_record(key, shape_list(array), dtype_name(array)))
    return real_report(path, "npz", "npz_root", datasets, attrs={})


def inspect_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    datasets: list[dict[str, Any]] = []

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                visit(f"{prefix}/{key}" if prefix else key, item)
        elif isinstance(value, list):
            datasets.append(dataset_record(prefix, infer_json_shape(value), "json_list"))
        else:
            datasets.append(dataset_record(prefix, None, type(value).__name__))

    visit("", payload)
    return real_report(path, "json", "json_root", datasets, attrs={})


def infer_json_shape(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    shape: list[int] = []
    current = value
    while isinstance(current, list):
        shape.append(len(current))
        current = current[0] if current else None
    return shape


def stringify_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def real_report(
    path: Path,
    file_format: str,
    trajectory_id: str,
    datasets: list[dict[str, Any]],
    attrs: dict[str, Any],
    attrs_by_path: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    time_lengths = sorted(
        {
            record["shape"][0]
            for record in datasets
            if record.get("shape") and record.get("time_axis") == 0
        }
    )
    return {
        **command_record(),
        "mode": "real",
        "mock": False,
        "source": str(path),
        "file_format": file_format,
        "trajectory_id": trajectory_id,
        "trajectory_keys": [record["path"] for record in datasets],
        "trajectory_attrs": attrs,
        "metadata_attrs": attrs_by_path or {},
        "datasets": datasets,
        "time_dimension_convention": {
            "axis": 0 if time_lengths else None,
            "observed_time_lengths": time_lengths,
            "meaning": "time axis inferred from first array dimension; see docs/LIBERO_ACTION_SEMANTICS.md for action alignment",
            "action_alignment": "processed LIBERO HDF5 uses action_to_current_obs convention per docs/LIBERO_ACTION_SEMANTICS.md",
        },
        "dataset_item_contract_preview": dataset_item_contract_preview(),
        "future_leakage_risks": future_leakage_risks(),
        "unresolved_questions": unresolved_questions(real_data=True),
    }


def inspect_real(path: Path, trajectory: str | None) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in {".hdf5", ".h5"}:
        return inspect_hdf5(path, trajectory)
    if suffix == ".npz":
        return inspect_npz(path)
    if suffix == ".json":
        return inspect_json(path)
    raise SystemExit(f"Unsupported file type: {path.suffix}. Use .hdf5, .h5, .npz, or .json.")


def write_report(report: dict[str, Any], output_dir: Path, report_format: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stem = f"{timestamp}_libero_data_inspection_{report['mode']}"
    if report_format == "json":
        path = output_dir / f"{stem}.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return path
    path = output_dir / f"{stem}.md"
    path.write_text(markdown_report(report), encoding="utf-8")
    return path


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# LIBERO Data Inspection",
        "",
        f"- Mode: `{report['mode']}`",
        f"- Mock: `{report['mock']}`",
        f"- Source: `{report['source']}`",
        f"- Trajectory: `{report['trajectory_id']}`",
        "",
        "## Datasets",
        "",
        "| Path | Shape | Dtype | Role | Source | Time Axis |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for record in report["datasets"]:
        lines.append(
            f"| `{record['path']}` | `{record['shape']}` | `{record['dtype']}` | `{record['field_role']}` | `{record.get('source_type', 'dataset')}` | `{record['time_axis']}` |"
        )
    lines.extend(["", "## Future Leakage Risks", ""])
    for risk in report["future_leakage_risks"]:
        lines.append(f"- {risk}")
    lines.extend(["", "## Unresolved Questions", ""])
    for question in report["unresolved_questions"]:
        lines.append(f"- {question}")
    return "\n".join(lines) + "\n"


def print_summary(report: dict[str, Any], output_path: Path) -> None:
    print(f"mode={report['mode']} mock={report['mock']}")
    print(f"source={report['source']}")
    print(f"trajectory_id={report['trajectory_id']}")
    print("trajectory_keys:")
    for key in report["trajectory_keys"]:
        print(f"  {key}")
    time_info = report["time_dimension_convention"]
    print(
        f"time_convention: axis={time_info.get('axis')} observed_time_lengths={time_info.get('observed_time_lengths')}"
    )
    print(f"action_alignment={time_info.get('action_alignment')}")
    print("datasets:")
    for record in report["datasets"]:
        print(
            f"  {record['path']}: shape={record['shape']} dtype={record['dtype']} role={record['field_role']} source={record.get('source_type', 'dataset')} time_axis={record['time_axis']}"
        )
    print(f"report={output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--path", type=Path, help="Path to one LIBERO demo file.")
    source.add_argument("--mock", action="store_true", help="Inspect a labeled mock trajectory.")
    parser.add_argument(
        "--trajectory",
        help="Optional HDF5 group path such as data/demo_0. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/inspections"),
        help="Directory for inspection reports. Default: results/inspections.",
    )
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    args = parser.parse_args()

    if args.mock:
        report = mock_report()
    else:
        if not args.path or not args.path.exists():
            raise SystemExit(f"Input path does not exist: {args.path}")
        report = inspect_real(args.path, args.trajectory)

    output_path = write_report(report, args.output_dir, args.format)
    print_summary(report, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
