#!/usr/bin/env python3
"""Diagnose LIBERO HDF5 action/observation alignment for one demonstration."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.trajectory_window import (  # noqa: E402
    ACTION_FROM_CURRENT_OBS,
    ACTION_TO_CURRENT_OBS,
    action_index_ranges,
    valid_time_indices,
)


DEFAULT_SUITE = "libero_spatial"


def find_demo_path(dataset_root: Path | None, suite: str) -> Path | None:
    if dataset_root is None:
        return None
    roots = [
        dataset_root / suite,
        dataset_root / f"{suite}_no_noops",
        dataset_root / "datasets" / suite,
        dataset_root,
    ]
    seen: set[Path] = set()
    for root in roots:
        root = root.expanduser()
        if root in seen or not root.exists():
            continue
        seen.add(root)
        matches = sorted(root.rglob("*.hdf5"))
        if matches:
            return matches[0]
    return None


def configured_dataset_root() -> Path | None:
    for env_name in ("LIBERO_DATASET_ROOT", "LIBERO_DATA_ROOT"):
        value = os.environ.get(env_name)
        if value:
            return Path(value)
    return None


def find_trajectory_group(handle: Any, requested: str | None) -> tuple[str, Any]:
    if requested:
        return requested, handle[requested]
    if "data" in handle:
        data = handle["data"]
        for key in sorted(data.keys()):
            if hasattr(data[key], "keys"):
                return f"data/{key}", data[key]
    raise SystemExit("No HDF5 trajectory group found. Expected data/demo_*.")


def relation(actions_len: int, images_len: int) -> str:
    if actions_len == images_len:
        return "len(actions) == len(images)"
    if actions_len == images_len - 1:
        return "len(actions) == len(images) - 1"
    if actions_len == images_len + 1:
        return "len(actions) == len(images) + 1"
    return f"other: len(actions)={actions_len}, len(images)={images_len}"


def inspect_alignment(
    path: Path,
    *,
    trajectory: str | None,
    convention: str,
    history_len: int,
    action_horizon: int,
    future_horizon: int,
    sample_count: int,
) -> dict[str, Any]:
    try:
        import h5py  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit("BLOCKER: h5py is required to inspect a real LIBERO HDF5 demo.") from exc

    with h5py.File(path, "r") as handle:
        trajectory_id, group = find_trajectory_group(handle, trajectory)
        if "actions" not in group or "obs" not in group:
            raise SystemExit("BLOCKER: trajectory must contain actions and obs groups.")
        obs_group = group["obs"]
        image_keys = [
            key
            for key in sorted(obs_group.keys())
            if "rgb" in key or "image" in key or "agentview" in key or "eye_in_hand" in key
        ]
        if not image_keys:
            raise SystemExit("BLOCKER: no image observation key found under obs.")
        primary_image_key = image_keys[0]
        actions = group["actions"]
        images = obs_group[primary_image_key]

        state_records = {}
        for key in ["states", "robot_states"]:
            if key in group:
                state_records[key] = {
                    "shape": list(group[key].shape),
                    "dtype": str(group[key].dtype),
                }
        for key in sorted(obs_group.keys()):
            if key not in image_keys:
                state_records[f"obs/{key}"] = {
                    "shape": list(obs_group[key].shape),
                    "dtype": str(obs_group[key].dtype),
                }

        valid_times = list(
            valid_time_indices(
                int(actions.shape[0]),
                history_len=history_len,
                action_horizon=action_horizon,
                future_horizon=future_horizon,
                action_convention=convention,
            )
        )
        if valid_times:
            selected = sorted(
                {
                    valid_times[0],
                    valid_times[len(valid_times) // 2],
                    valid_times[-1],
                }
            )[:sample_count]
        else:
            selected = []

        candidates = []
        for t in selected:
            h_start, h_stop, target_start, target_stop = action_index_ranges(
                t,
                history_len=history_len,
                action_horizon=action_horizon,
                action_convention=convention,
            )
            future_start = t + 1
            future_stop = future_start + future_horizon
            candidates.append(
                {
                    "t": t,
                    "image_t_index": t,
                    "action_history_indices": list(range(h_start, h_stop)),
                    "target_action_indices": list(range(target_start, target_stop)),
                    "future_frame_indices": list(range(future_start, future_stop)),
                    "target_actions_consistent_with_convention": (
                        target_start == t + 1
                        if convention == ACTION_TO_CURRENT_OBS
                        else target_start == t
                    ),
                }
            )

        return {
            "path": str(path),
            "trajectory_id": trajectory_id,
            "primary_image_key": f"obs/{primary_image_key}",
            "actions": {"shape": list(actions.shape), "dtype": str(actions.dtype)},
            "image": {"shape": list(images.shape), "dtype": str(images.dtype)},
            "states": state_records,
            "length_relation": relation(int(actions.shape[0]), int(images.shape[0])),
            "first_valid_index": valid_times[0] if valid_times else None,
            "last_valid_index": valid_times[-1] if valid_times else None,
            "convention": convention,
            "convention_summary": (
                "actions[t] led to obs[t]; target actions after image_t start at t+1"
                if convention == ACTION_TO_CURRENT_OBS
                else "actions[t] executes after obs[t]; target actions after image_t start at t"
            ),
            "candidate_supervised_samples": candidates,
        }


def print_report(report: dict[str, Any]) -> None:
    print(f"path={report['path']}")
    print(f"trajectory_id={report['trajectory_id']}")
    print(f"primary_image_key={report['primary_image_key']}")
    print(f"actions_shape={report['actions']['shape']} dtype={report['actions']['dtype']}")
    print(f"image_shape={report['image']['shape']} dtype={report['image']['dtype']}")
    for key, record in report["states"].items():
        print(f"{key}_shape={record['shape']} dtype={record['dtype']}")
    print(f"length_relation={report['length_relation']}")
    print(f"first_valid_index={report['first_valid_index']}")
    print(f"last_valid_index={report['last_valid_index']}")
    print(f"convention={report['convention']}")
    print(f"convention_summary={report['convention_summary']}")
    print("candidate_supervised_samples:")
    for sample in report["candidate_supervised_samples"]:
        print(
            "  "
            f"t={sample['t']} image_t={sample['image_t_index']} "
            f"history={sample['action_history_indices']} "
            f"target_actions={sample['target_action_indices']} "
            f"future_frames={sample['future_frame_indices']} "
            f"consistent={sample['target_actions_consistent_with_convention']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, help="Path to one LIBERO HDF5 demo.")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument("--trajectory", help="HDF5 trajectory group such as data/demo_0.")
    parser.add_argument("--history-len", type=int, default=4)
    parser.add_argument("--action-horizon", type=int, default=4)
    parser.add_argument("--future-horizon", type=int, default=4)
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument(
        "--convention",
        choices=[ACTION_TO_CURRENT_OBS, ACTION_FROM_CURRENT_OBS],
        default=ACTION_TO_CURRENT_OBS,
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    path = args.path
    if path is None:
        path = find_demo_path(args.dataset_root or configured_dataset_root(), args.suite)
    if path is None or not path.exists():
        print(
            "BLOCKER: no real LIBERO HDF5 demo found. Provide --path or set LIBERO_DATASET_ROOT.",
            file=sys.stderr,
        )
        return 2

    report = inspect_alignment(
        path,
        trajectory=args.trajectory,
        convention=args.convention,
        history_len=args.history_len,
        action_horizon=args.action_horizon,
        future_horizon=args.future_horizon,
        sample_count=args.sample_count,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
