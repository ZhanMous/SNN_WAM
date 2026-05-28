#!/usr/bin/env python3
"""Dry-check or run one tiny offscreen LIBERO env reset/step smoke."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

from _common import run_git  # noqa: E402


def import_status(module_name: str) -> dict[str, Any]:
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, AttributeError, ValueError) as exc:
        return {"available": False, "error": repr(exc), "origin": None}
    if spec is None:
        return {"available": False, "error": None, "origin": None}
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - depends on LIBERO install.
        return {"available": False, "error": repr(exc), "origin": spec.origin}
    return {
        "available": True,
        "error": None,
        "origin": spec.origin,
        "version": getattr(module, "__version__", None),
    }


def zero_action(env: Any) -> Any:
    try:
        import numpy as np  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on optional package.
        raise RuntimeError("NumPy is required to construct a zero action.") from exc

    action_space = getattr(env, "action_space", None)
    if action_space is not None and getattr(action_space, "shape", None):
        return np.zeros(action_space.shape, dtype="float32")

    action_spec = getattr(env, "action_spec", None)
    if action_spec is not None:
        spec_value = action_spec() if callable(action_spec) else action_spec
        if isinstance(spec_value, tuple) and len(spec_value) >= 1:
            return np.zeros_like(spec_value[0], dtype="float32")

    raise RuntimeError(
        "Could not infer action shape from env.action_space or env.action_spec."
    )


def task_name(task: Any, task_id: int) -> str:
    for attr in ("language", "task_name", "name"):
        value = getattr(task, attr, None)
        if value:
            return str(value)
    return f"task_{task_id}"


def task_bddl_file(task: Any) -> str:
    for attr in ("bddl_file", "bddl_file_name"):
        value = getattr(task, attr, None)
        if value:
            return str(value)
    raise RuntimeError("Could not find task.bddl_file or task.bddl_file_name.")


def build_env_step_report(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", args.mujoco_gl)
    status_short = run_git(["status", "--short"])
    report: dict[str, Any] = {
        "smoke": "libero_env_step",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "settings": {
            "suite": args.suite,
            "task_id": args.task_id,
            "max_steps": args.max_steps,
            "run_step": args.run_step,
            "mujoco_gl": os.environ.get("MUJOCO_GL"),
        },
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
        "environment": {
            "LIBERO_DATA_ROOT": os.environ.get("LIBERO_DATA_ROOT"),
            "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
            "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM"),
        },
        "imports": {
            "libero": import_status("libero"),
            "libero.libero.benchmark": import_status("libero.libero.benchmark"),
            "libero.libero.envs": import_status("libero.libero.envs"),
        },
        "status": "not_started",
        "steps": [],
    }

    missing = [
        name
        for name, record in report["imports"].items()
        if not record.get("available", False)
    ]
    if missing:
        report["status"] = "missing_imports"
        report["missing_imports"] = missing
        return report

    from libero.libero import benchmark  # type: ignore[import-not-found]
    from libero.libero.envs import OffScreenRenderEnv  # type: ignore[import-not-found]

    benchmark_dict = benchmark.get_benchmark_dict()
    if args.suite not in benchmark_dict:
        report["status"] = "invalid_suite"
        report["available_suites"] = sorted(benchmark_dict)
        return report

    suite = benchmark_dict[args.suite]()
    task = suite.get_task(args.task_id)
    report["task"] = {
        "task_id": args.task_id,
        "task_name": task_name(task, args.task_id),
        "bddl_file": task_bddl_file(task),
    }

    if not args.run_step:
        report["status"] = "dry_run_ok"
        report["note"] = "No environment was instantiated; pass --run-step to reset/step."
        return report

    env = None
    try:
        env = OffScreenRenderEnv(
            bddl_file_name=report["task"]["bddl_file"],
            camera_heights=args.camera_height,
            camera_widths=args.camera_width,
        )
        if hasattr(env, "seed"):
            env.seed(args.seed)
        reset_obs = env.reset()
        report["reset_observation_type"] = type(reset_obs).__name__

        for step_id in range(args.max_steps):
            action = zero_action(env)
            step_result = env.step(action)
            if isinstance(step_result, tuple):
                report["steps"].append(
                    {
                        "step_id": step_id,
                        "tuple_len": len(step_result),
                        "result_types": [type(value).__name__ for value in step_result],
                    }
                )
            else:
                report["steps"].append(
                    {"step_id": step_id, "result_type": type(step_result).__name__}
                )
        report["status"] = "step_ok"
    finally:
        if env is not None and hasattr(env, "close"):
            env.close()
    return report


def write_log(report: dict[str, Any], log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"{timestamp}_libero_env_step.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_spatial", help="LIBERO suite name.")
    parser.add_argument("--task-id", type=int, default=0, help="LIBERO task index.")
    parser.add_argument("--seed", type=int, default=0, help="Env seed for smoke run.")
    parser.add_argument("--max-steps", type=int, default=1, help="Tiny smoke step count.")
    parser.add_argument(
        "--run-step",
        action="store_true",
        help="Actually instantiate OffScreenRenderEnv, reset, and step. Without this flag, only imports/task API are checked.",
    )
    parser.add_argument(
        "--mujoco-gl",
        default=os.environ.get("MUJOCO_GL", "egl"),
        help="Offscreen MuJoCo backend to set when unset. Default: env MUJOCO_GL or egl.",
    )
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("results/smoke"),
        help="Directory for JSON smoke logs. Default: results/smoke.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Exit 0 when LIBERO imports are unavailable.",
    )
    args = parser.parse_args()

    try:
        report = build_env_step_report(args)
    except Exception as exc:  # pragma: no cover - depends on LIBERO runtime.
        report = {
            "smoke": "libero_env_step",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "command": sys.argv,
            "status": "error",
            "error": repr(exc),
        }

    log_path = write_log(report, args.log_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"smoke_log={log_path}")

    if report["status"] in {"dry_run_ok", "step_ok"}:
        return 0
    if report["status"] == "missing_imports" and args.allow_missing:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
