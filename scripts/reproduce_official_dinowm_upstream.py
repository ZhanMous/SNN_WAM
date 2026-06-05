#!/usr/bin/env python3
"""Prepare and optionally run the official upstream DINO-WM reproduction.

This script targets the upstream repository at
https://github.com/gaoyuezhou/dino_wm. By default it performs preflight checks
and writes a reproducibility package with the exact train/plan commands. It only
executes upstream training or planning when ``--execute`` is passed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


OFFICIAL_REPO_URL = "https://github.com/gaoyuezhou/dino_wm"
OFFICIAL_DATA_URL = "https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28"
OFFICIAL_PROJECT_URL = "https://dino-wm.github.io/"
OFFICIAL_PAPER_URL = "https://arxiv.org/abs/2411.04983"

ENV_DATA_SUBDIRS = {
    "point_maze": "point_maze",
    "pusht": "pusht_noise",
}


@dataclass(frozen=True)
class ReproCommands:
    train: list[str]
    plan: list[str]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream_dir",
        type=Path,
        required=True,
        help="Path to a clone of https://github.com/gaoyuezhou/dino_wm.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path(os.environ["DATASET_DIR"]) if "DATASET_DIR" in os.environ else None,
        help="Official DINO-WM DATASET_DIR containing point_maze/pusht_noise.",
    )
    parser.add_argument(
        "--artifact_dir",
        type=Path,
        default=None,
        help="Where to write the upstream reproduction package.",
    )
    parser.add_argument("--env", choices=sorted(ENV_DATA_SUBDIRS), default="point_maze")
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--num_hist", type=int, default=3)
    parser.add_argument("--num_pred", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ckpt_base_path",
        type=Path,
        default=None,
        help="ckpt_base_path override for official Hydra configs.",
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help="Deterministic official output model_name under ckpt_base_path/outputs.",
    )
    parser.add_argument(
        "--python_exe",
        default=sys.executable,
        help="Python executable inside the official dino_wm environment.",
    )
    parser.add_argument(
        "--wandb_mode",
        choices=["online", "offline", "disabled"],
        default="offline",
        help="WANDB_MODE to use when executing upstream train/plan commands.",
    )
    parser.add_argument(
        "--stage",
        choices=["preflight", "train", "plan", "all"],
        default="preflight",
        help="Stage to prepare or execute.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run the official upstream command. Without this, only write artifacts.",
    )
    parser.add_argument("--n_evals", type=int, default=5)
    parser.add_argument("--planner", default="cem")
    parser.add_argument("--goal_H", type=int, default=5)
    parser.add_argument("--goal_source", default="random_state")
    parser.add_argument("--planner_opt_steps", type=int, default=30)
    parser.add_argument(
        "--extra_train_arg",
        action="append",
        default=[],
        help="Additional Hydra override for train.py, e.g. training.epochs=1.",
    )
    parser.add_argument(
        "--extra_plan_arg",
        action="append",
        default=[],
        help="Additional Hydra override for plan.py.",
    )
    return parser.parse_args(argv)


def default_artifact_dir(env_name: str, seed: int) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path("results/upstream") / f"{stamp}_official_dinowm_{env_name}_seed{seed}"


def normalize_paths(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve paths before executing commands from the upstream repo cwd."""
    args.upstream_dir = args.upstream_dir.expanduser().resolve()
    if args.dataset_dir is not None:
        args.dataset_dir = args.dataset_dir.expanduser().resolve()
    args.artifact_dir = args.artifact_dir.expanduser().resolve()
    if args.ckpt_base_path is not None:
        args.ckpt_base_path = args.ckpt_base_path.expanduser().resolve()
    return args


def resolve_run_names(args: argparse.Namespace) -> tuple[Path, str]:
    ckpt_base_path = args.ckpt_base_path or (args.artifact_dir / "official_ckpts")
    model_name = args.model_name or (
        f"{args.env}_official_frameskip{args.frameskip}_hist{args.num_hist}_seed{args.seed}"
    )
    return ckpt_base_path, model_name


def build_commands(args: argparse.Namespace) -> ReproCommands:
    ckpt_base_path, model_name = resolve_run_names(args)
    output_dir = ckpt_base_path / "outputs" / model_name

    train_cmd = [
        args.python_exe,
        "train.py",
        "--config-name",
        "train.yaml",
        f"env={args.env}",
        f"frameskip={args.frameskip}",
        f"num_hist={args.num_hist}",
        f"num_pred={args.num_pred}",
        f"training.seed={args.seed}",
        f"ckpt_base_path={ckpt_base_path}",
        f"hydra.run.dir={output_dir}",
        f"hydra.sweep.dir={output_dir}",
        *args.extra_train_arg,
    ]

    plan_cmd = [
        args.python_exe,
        "plan.py",
        f"model_name={model_name}",
        f"n_evals={args.n_evals}",
        f"planner={args.planner}",
        f"goal_H={args.goal_H}",
        f"goal_source={args.goal_source}",
        f"planner.opt_steps={args.planner_opt_steps}",
        f"ckpt_base_path={ckpt_base_path}",
        f"seed={args.seed}",
        *args.extra_plan_arg,
    ]

    return ReproCommands(train=train_cmd, plan=plan_cmd)


def preflight_checks(args: argparse.Namespace) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    upstream_dir = args.upstream_dir
    add("upstream_dir_exists", upstream_dir.exists(), str(upstream_dir))
    for rel_path in ["train.py", "plan.py", "conf/train.yaml", "conf/plan.yaml"]:
        path = upstream_dir / rel_path
        add(f"upstream_has_{rel_path}", path.exists(), str(path))

    if args.dataset_dir is None:
        add("dataset_dir_set", False, "DATASET_DIR is not set and --dataset_dir was not passed")
    else:
        add("dataset_dir_exists", args.dataset_dir.exists(), str(args.dataset_dir))
        env_subdir = args.dataset_dir / ENV_DATA_SUBDIRS[args.env]
        add(f"dataset_has_{ENV_DATA_SUBDIRS[args.env]}", env_subdir.exists(), str(env_subdir))

    return checks


def get_git_commit(repo_dir: Path) -> str | None:
    """Return HEAD commit for a git repo, or None if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def capture_environment(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host_python": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "target_python_exe": args.python_exe,
        "upstream_dir": str(args.upstream_dir),
        "dataset_dir": str(args.dataset_dir) if args.dataset_dir is not None else None,
        "upstream_commit": get_git_commit(args.upstream_dir),
        "official_repo_url": OFFICIAL_REPO_URL,
        "official_data_url": OFFICIAL_DATA_URL,
    }


def command_to_shell(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def write_repro_package(
    artifact_dir: Path,
    *,
    args: argparse.Namespace,
    commands: ReproCommands,
    checks: list[dict[str, Any]],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)

    command_lines = [
        "# Official upstream DINO-WM reproduction commands",
        f"# repo: {OFFICIAL_REPO_URL}",
        f"# data: {OFFICIAL_DATA_URL}",
        "",
        "export DATASET_DIR="
        + shlex.quote(str(args.dataset_dir) if args.dataset_dir is not None else "/path/to/data"),
        "export WANDB_MODE=" + shlex.quote(args.wandb_mode),
        "",
        "# Train official DINO-WM",
        command_to_shell(commands.train),
        "",
        "# Plan with the trained official DINO-WM",
        command_to_shell(commands.plan),
        "",
    ]
    (artifact_dir / "command.sh").write_text("\n".join(command_lines))

    payload = {
        "status": "prepared" if all(c["ok"] for c in checks) else "preflight_failed",
        "stage": args.stage,
        "execute": args.execute,
        "env": args.env,
        "frameskip": args.frameskip,
        "num_hist": args.num_hist,
        "num_pred": args.num_pred,
        "seed": args.seed,
        "python_exe": args.python_exe,
        "wandb_mode": args.wandb_mode,
        "model_name": resolve_run_names(args)[1],
        "upstream_commit": get_git_commit(args.upstream_dir),
        "extra_train_arg": args.extra_train_arg,
        "extra_plan_arg": args.extra_plan_arg,
        "checks": checks,
        "commands": {
            "train": commands.train,
            "plan": commands.plan,
        },
    }
    (artifact_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    (artifact_dir / "environment.txt").write_text(
        json.dumps(capture_environment(args), indent=2, default=str) + "\n"
    )
    (artifact_dir / "sources.json").write_text(
        json.dumps(
            {
                "official_repo": OFFICIAL_REPO_URL,
                "official_data": OFFICIAL_DATA_URL,
                "official_project": OFFICIAL_PROJECT_URL,
                "official_paper": OFFICIAL_PAPER_URL,
            },
            indent=2,
        )
        + "\n"
    )
    (artifact_dir / "notes.md").write_text(
        "# Official DINO-WM Upstream Reproduction\n\n"
        "This package records commands for the unmodified upstream DINO-WM repo. "
        "It is not evidence of reproduction until `summary.json` is accompanied by "
        "official training metrics/checkpoints and planning outputs from `plan.py`.\n\n"
        "Required success evidence:\n"
        "- official repo commit\n"
        "- DATASET_DIR with official task data\n"
        "- train.py logs/checkpoint under ckpt_base_path/outputs/<model_name>\n"
        "- plan.py outputs under upstream plan_outputs\n"
    )


def update_execution_status(
    artifact_dir: Path,
    execution_results: list[dict[str, Any]],
) -> None:
    summary_path = artifact_dir / "summary.json"
    payload = json.loads(summary_path.read_text())
    payload["execution_results"] = execution_results
    payload["status"] = (
        "executed" if all(result["return_code"] == 0 for result in execution_results)
        else "execution_failed"
    )
    summary_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def execute_command(cmd: Sequence[str], *, cwd: Path, env: dict[str, str]) -> int:
    print(command_to_shell(cmd), flush=True)
    completed = subprocess.run(list(cmd), cwd=str(cwd), env=env)
    return int(completed.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.artifact_dir is None:
        args.artifact_dir = default_artifact_dir(args.env, args.seed)
    args = normalize_paths(args)

    commands = build_commands(args)
    checks = preflight_checks(args)
    write_repro_package(args.artifact_dir, args=args, commands=commands, checks=checks)

    failed = [check for check in checks if not check["ok"]]
    if failed:
        print(f"Preflight failed: {len(failed)} issue(s). See {args.artifact_dir / 'summary.json'}")
        for check in failed:
            print(f"  - {check['name']}: {check['detail']}")
        return 2

    if not args.execute or args.stage == "preflight":
        print(f"Prepared official DINO-WM reproduction package: {args.artifact_dir}")
        return 0

    env = dict(os.environ)
    if args.dataset_dir is not None:
        env["DATASET_DIR"] = str(args.dataset_dir)
    env["WANDB_MODE"] = args.wandb_mode

    stages = ["train", "plan"] if args.stage == "all" else [args.stage]
    execution_results: list[dict[str, Any]] = []
    for stage in stages:
        cmd = commands.train if stage == "train" else commands.plan
        rc = execute_command(cmd, cwd=args.upstream_dir, env=env)
        execution_results.append({"stage": stage, "return_code": rc})
        update_execution_status(args.artifact_dir, execution_results)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
