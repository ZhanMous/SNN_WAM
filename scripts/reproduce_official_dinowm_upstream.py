#!/usr/bin/env python3
"""Prepare and optionally run the official upstream DINO-WM reproduction.

This script targets the upstream repository at
https://github.com/gaoyuezhou/dino_wm. By default it performs preflight checks
and writes a reproducibility package with the exact train/plan commands. It only
executes upstream training or planning when ``--execute`` is passed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shlex
import shutil
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
TRAIN_LOSS_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)\s+Training loss:\s+"
    r"(?P<train_loss>[0-9.eE+-]+)\s+Validation loss:\s+(?P<val_loss>[0-9.eE+-]+)"
)
PLAN_OUTPUT_RE = re.compile(r"Planning result saved dir:\s*(?P<path>\S+)")

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
        "--torch_home",
        type=Path,
        default=None,
        help="Artifact-local TORCH_HOME to use for torch.hub caches and checkpoints.",
    )
    parser.add_argument(
        "--dinov2_github_ref",
        default=None,
        help="DINOv2 GitHub commit/ref recorded for torch.hub reproducibility.",
    )
    parser.add_argument(
        "--dinov2_cache_source",
        type=Path,
        default=None,
        help="Existing DINOv2 source tree to install as TORCH_HOME/hub/facebookresearch_dinov2_main.",
    )
    parser.add_argument(
        "--dinov2_checkpoint_source",
        type=Path,
        default=None,
        help="Optional DINOv2 checkpoint file to copy into TORCH_HOME/hub/checkpoints.",
    )
    parser.add_argument(
        "--pin_dinov2_main_cache",
        action="store_true",
        help="Install --dinov2_cache_source as artifact-local torch.hub main cache.",
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
    parser.add_argument(
        "--extra_env",
        action="append",
        default=[],
        help="Additional environment variable exported for train/plan execution, as KEY=VALUE.",
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
    if args.torch_home is not None:
        args.torch_home = args.torch_home.expanduser().resolve()
    if args.dinov2_cache_source is not None:
        args.dinov2_cache_source = args.dinov2_cache_source.expanduser().resolve()
    if args.dinov2_checkpoint_source is not None:
        args.dinov2_checkpoint_source = args.dinov2_checkpoint_source.expanduser().resolve()
    return args


def resolve_run_names(args: argparse.Namespace) -> tuple[Path, str]:
    ckpt_base_path = args.ckpt_base_path or (args.artifact_dir / "official_ckpts")
    model_name = args.model_name or (
        f"{args.env}_official_frameskip{args.frameskip}_hist{args.num_hist}_seed{args.seed}"
    )
    return ckpt_base_path, model_name


def official_output_dir(args: argparse.Namespace) -> Path:
    ckpt_base_path, model_name = resolve_run_names(args)
    return ckpt_base_path / "outputs" / model_name


def build_commands(args: argparse.Namespace) -> ReproCommands:
    ckpt_base_path, model_name = resolve_run_names(args)
    output_dir = official_output_dir(args)

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

    if args.pin_dinov2_main_cache:
        if args.torch_home is None:
            add("torch_home_set_for_dinov2_pin", False, "--torch_home is required")
        else:
            cache_dir = args.torch_home / "hub" / "facebookresearch_dinov2_main"
            add("dinov2_main_cache_installed", cache_dir.exists(), str(cache_dir))
        add(
            "dinov2_github_ref_set",
            bool(args.dinov2_github_ref),
            args.dinov2_github_ref or "--dinov2_github_ref is required",
        )

    return checks


def path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def prepare_dinov2_main_cache(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Install a pinned DINOv2 tree as artifact-local torch.hub main cache."""
    if not args.pin_dinov2_main_cache:
        return []
    if args.torch_home is None:
        raise ValueError("--torch_home is required with --pin_dinov2_main_cache")
    if args.dinov2_cache_source is None:
        raise ValueError("--dinov2_cache_source is required with --pin_dinov2_main_cache")
    if not args.dinov2_github_ref:
        raise ValueError("--dinov2_github_ref is required with --pin_dinov2_main_cache")
    if not path_is_relative_to(args.torch_home, args.artifact_dir):
        raise ValueError("--torch_home must be inside --artifact_dir when pinning DINOv2 cache")
    if not args.dinov2_cache_source.exists():
        raise FileNotFoundError(args.dinov2_cache_source)

    hub_dir = args.torch_home / "hub"
    dest = hub_dir / "facebookresearch_dinov2_main"
    if dest.exists():
        shutil.rmtree(dest)
    hub_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        args.dinov2_cache_source,
        dest,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (dest / "DINOV2_GITHUB_REF.txt").write_text(args.dinov2_github_ref + "\n")

    records: list[dict[str, Any]] = [
        {
            "name": "dinov2_main_cache",
            "source": str(args.dinov2_cache_source),
            "destination": str(dest),
            "github_ref": args.dinov2_github_ref,
        }
    ]

    if args.dinov2_checkpoint_source is not None:
        if not args.dinov2_checkpoint_source.exists():
            raise FileNotFoundError(args.dinov2_checkpoint_source)
        checkpoint_dir = hub_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dest = checkpoint_dir / args.dinov2_checkpoint_source.name
        shutil.copy2(args.dinov2_checkpoint_source, checkpoint_dest)
        records.append(
            {
                "name": "dinov2_checkpoint",
                "source": str(args.dinov2_checkpoint_source),
                "destination": str(checkpoint_dest),
            }
        )

    return records


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
        "target_torch_home": str(args.torch_home) if args.torch_home is not None else None,
        "dinov2_github_ref": args.dinov2_github_ref,
        "upstream_dir": str(args.upstream_dir),
        "dataset_dir": str(args.dataset_dir) if args.dataset_dir is not None else None,
        "upstream_commit": get_git_commit(args.upstream_dir),
        "official_repo_url": OFFICIAL_REPO_URL,
        "official_data_url": OFFICIAL_DATA_URL,
    }


def command_to_shell(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def parse_extra_env(values: Sequence[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        key, sep, env_value = value.partition("=")
        if not sep or not key:
            raise ValueError(f"Invalid --extra_env value: {value!r}; expected KEY=VALUE")
        parsed[key] = env_value
    return parsed


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
    ]
    if args.torch_home is not None:
        command_lines.append("export TORCH_HOME=" + shlex.quote(str(args.torch_home)))
    for key, value in parse_extra_env(args.extra_env).items():
        command_lines.append(f"export {key}=" + shlex.quote(value))
    command_lines.extend(
        [
            "cd " + shlex.quote(str(args.upstream_dir)),
            "",
            "# Train official DINO-WM",
            command_to_shell(commands.train),
            "",
            "# Plan with the trained official DINO-WM",
            command_to_shell(commands.plan),
            "",
        ]
    )
    command_text = "\n".join(command_lines)
    (artifact_dir / "command.sh").write_text(command_text)
    (artifact_dir / "command.txt").write_text(command_text)

    config_payload = {
        "description": "Wrapper config for unmodified upstream DINO-WM reproduction.",
        "upstream_dir": str(args.upstream_dir),
        "dataset_dir": str(args.dataset_dir) if args.dataset_dir is not None else None,
        "artifact_dir": str(artifact_dir),
        "stage": args.stage,
        "execute": args.execute,
        "env": args.env,
        "frameskip": args.frameskip,
        "num_hist": args.num_hist,
        "num_pred": args.num_pred,
        "seed": args.seed,
        "ckpt_base_path": str(resolve_run_names(args)[0]),
        "model_name": resolve_run_names(args)[1],
        "python_exe": args.python_exe,
        "torch_home": str(args.torch_home) if args.torch_home is not None else None,
        "dinov2_github_ref": args.dinov2_github_ref,
        "wandb_mode": args.wandb_mode,
        "n_evals": args.n_evals,
        "planner": args.planner,
        "goal_H": args.goal_H,
        "goal_source": args.goal_source,
        "planner_opt_steps": args.planner_opt_steps,
        "extra_train_arg": args.extra_train_arg,
        "extra_plan_arg": args.extra_plan_arg,
        "extra_env": args.extra_env,
        "commands": {
            "train": commands.train,
            "plan": commands.plan,
        },
    }
    # JSON is valid YAML 1.2 and avoids adding a PyYAML dependency to the helper.
    (artifact_dir / "config.yaml").write_text(json.dumps(config_payload, indent=2) + "\n")

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
        "torch_home": str(args.torch_home) if args.torch_home is not None else None,
        "dinov2_github_ref": args.dinov2_github_ref,
        "pin_dinov2_main_cache": args.pin_dinov2_main_cache,
        "wandb_mode": args.wandb_mode,
        "model_name": resolve_run_names(args)[1],
        "upstream_commit": get_git_commit(args.upstream_dir),
        "extra_train_arg": args.extra_train_arg,
        "extra_plan_arg": args.extra_plan_arg,
        "extra_env": args.extra_env,
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
    (artifact_dir / "git_commit.txt").write_text(
        f"upstream_repo={args.upstream_dir}\n"
        f"upstream_commit={get_git_commit(args.upstream_dir)}\n"
        f"wrapper_repo={Path.cwd()}\n"
        f"wrapper_commit={get_git_commit(Path.cwd())}\n"
    )
    (artifact_dir / "seeds.txt").write_text(f"training.seed={args.seed}\n")
    (artifact_dir / "split.json").write_text(
        json.dumps(
            {
                "env": args.env,
                "dataset_dir": str(args.dataset_dir) if args.dataset_dir is not None else None,
                "official_split": "defined by upstream dataset loader/config",
                "notes": "For smoke overrides, inspect extra_train_arg in summary.json.",
            },
            indent=2,
        )
        + "\n"
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


def refresh_artifact_link(link_path: Path, target_path: Path) -> None:
    """Create a stable artifact pointer to a large upstream-produced file."""
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    try:
        relative_target = os.path.relpath(target_path, link_path.parent)
        link_path.symlink_to(relative_target)
    except OSError:
        shutil.copy2(target_path, link_path)


def copy_if_exists(source: Path, destination: Path) -> str | None:
    if not source.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return str(destination)


def collect_train_artifacts(artifact_dir: Path, args: argparse.Namespace) -> dict[str, str]:
    output_dir = official_output_dir(args)
    records: dict[str, str] = {"official_output_dir": str(output_dir)}

    config_path = copy_if_exists(output_dir / ".hydra" / "config.yaml", artifact_dir / "official_train_config.yaml")
    if config_path is not None:
        records["official_train_config"] = config_path
    overrides_path = copy_if_exists(
        output_dir / ".hydra" / "overrides.yaml",
        artifact_dir / "official_train_overrides.yaml",
    )
    if overrides_path is not None:
        records["official_train_overrides"] = overrides_path

    checkpoint_source = output_dir / "checkpoints" / "model_latest.pth"
    if checkpoint_source.exists():
        checkpoint_link = artifact_dir / "checkpoint.pt"
        refresh_artifact_link(checkpoint_link, checkpoint_source)
        records["checkpoint"] = str(checkpoint_link)
        records["checkpoint_source"] = str(checkpoint_source)

    return records


def find_latest_plan_output_dir(args: argparse.Namespace, log_path: Path | None = None) -> Path | None:
    if log_path is not None and log_path.exists():
        for line in reversed(log_path.read_text(errors="replace").splitlines()):
            match = PLAN_OUTPUT_RE.search(line)
            if match:
                path = Path(match.group("path"))
                if path.exists():
                    return path

    _, model_name = resolve_run_names(args)
    suffix = f"_{model_name}_gH{args.goal_H}"
    plan_outputs = args.upstream_dir / "plan_outputs"
    if not plan_outputs.exists():
        return None
    candidates = [
        path
        for path in plan_outputs.iterdir()
        if path.is_dir() and path.name.endswith(suffix)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def collect_plan_artifacts(
    artifact_dir: Path,
    args: argparse.Namespace,
    log_path: Path | None = None,
) -> dict[str, str]:
    records: dict[str, str] = {}
    plan_output_dir = find_latest_plan_output_dir(args, log_path)
    if plan_output_dir is None:
        return records

    records["plan_output_dir"] = str(plan_output_dir)
    (artifact_dir / "plan_outputs.txt").write_text(str(plan_output_dir) + "\n")

    config_path = copy_if_exists(
        plan_output_dir / ".hydra" / "config.yaml",
        artifact_dir / "official_plan_config.yaml",
    )
    if config_path is not None:
        records["official_plan_config"] = config_path
    overrides_path = copy_if_exists(
        plan_output_dir / ".hydra" / "overrides.yaml",
        artifact_dir / "official_plan_overrides.yaml",
    )
    if overrides_path is not None:
        records["official_plan_overrides"] = overrides_path

    return records


def write_train_metrics_csv(artifact_dir: Path, args: argparse.Namespace) -> Path | None:
    train_log = official_output_dir(args) / "train.log"
    if not train_log.exists():
        return None

    rows = []
    for line in train_log.read_text().splitlines():
        match = TRAIN_LOSS_RE.search(line)
        if match:
            rows.append(
                {
                    "epoch": int(match.group("epoch")),
                    "train_loss": float(match.group("train_loss")),
                    "val_loss": float(match.group("val_loss")),
                    "source_log": str(train_log),
                }
            )
    if not rows:
        return None

    artifact_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = artifact_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return metrics_path


def write_plan_metrics_csv(
    artifact_dir: Path,
    *,
    args: argparse.Namespace,
    plan_output_dir: Path | None,
    log_path: Path | None,
) -> Path | None:
    metrics: dict[str, Any] = {}
    logs_json = plan_output_dir / "logs.json" if plan_output_dir is not None else None
    if logs_json is not None and logs_json.exists():
        for line in logs_json.read_text().splitlines():
            if not line.strip():
                continue
            try:
                metrics.update(json.loads(line))
            except json.JSONDecodeError:
                continue

    def metric_value(*keys: str) -> Any:
        for key in keys:
            if key in metrics:
                return metrics[key]
        return None

    success_rate = metric_value("final_eval/success_rate", "plan_0/success_rate")
    state_dist = metric_value("final_eval/mean_state_dist", "plan_0/mean_state_dist")
    visual_dist = metric_value("final_eval/mean_visual_dist", "plan_0/mean_visual_dist")
    proprio_dist = metric_value("final_eval/mean_proprio_dist", "plan_0/mean_proprio_dist")
    plan_loss = metric_value("plan_0/loss")
    if success_rate is None and log_path is not None and log_path.exists():
        for line in log_path.read_text(errors="replace").splitlines():
            if "Success rate:" in line:
                try:
                    success_rate = float(line.rsplit(":", 1)[1].strip())
                except ValueError:
                    pass

    if success_rate is None and state_dist is None and plan_loss is None:
        return None

    row = {
        "stage": "plan",
        "success_rate": success_rate,
        "state_dist": state_dist,
        "visual_dist": visual_dist,
        "proprio_dist": proprio_dist,
        "plan_loss": plan_loss,
        "source_log": str(log_path) if log_path is not None else None,
        "plan_output_dir": str(plan_output_dir) if plan_output_dir is not None else None,
    }

    metrics_path = artifact_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return metrics_path


def execute_command(
    cmd: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path | None = None,
) -> int:
    print(command_to_shell(cmd), flush=True)
    if log_path is None:
        completed = subprocess.run(list(cmd), cwd=str(cwd), env=env)
        return int(completed.returncode)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as handle:
        handle.write(command_to_shell(cmd) + "\n")
        handle.flush()
        completed = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert completed.stdout is not None
        for line in completed.stdout:
            print(line, end="")
            handle.write(line)
        return int(completed.wait())


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.artifact_dir is None:
        args.artifact_dir = default_artifact_dir(args.env, args.seed)
    args = normalize_paths(args)

    prepared_caches = prepare_dinov2_main_cache(args)
    commands = build_commands(args)
    checks = preflight_checks(args)
    write_repro_package(args.artifact_dir, args=args, commands=commands, checks=checks)
    if prepared_caches:
        cache_path = args.artifact_dir / "prepared_caches.json"
        cache_path.write_text(json.dumps(prepared_caches, indent=2, default=str) + "\n")

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
    if args.torch_home is not None:
        env["TORCH_HOME"] = str(args.torch_home)
    env.update(parse_extra_env(args.extra_env))

    stages = ["train", "plan"] if args.stage == "all" else [args.stage]
    execution_results: list[dict[str, Any]] = []
    for stage in stages:
        cmd = commands.train if stage == "train" else commands.plan
        log_path = args.artifact_dir / f"{stage}.log"
        rc = execute_command(cmd, cwd=args.upstream_dir, env=env, log_path=log_path)
        execution_results.append({"stage": stage, "return_code": rc, "log": str(log_path)})
        if stage == "train" and rc == 0:
            metrics_path = write_train_metrics_csv(args.artifact_dir, args)
            if metrics_path is not None:
                execution_results[-1]["metrics_csv"] = str(metrics_path)
            execution_results[-1].update(collect_train_artifacts(args.artifact_dir, args))
        if stage == "plan" and rc == 0:
            plan_records = collect_plan_artifacts(args.artifact_dir, args, log_path)
            execution_results[-1].update(plan_records)
            plan_output_dir = (
                Path(plan_records["plan_output_dir"])
                if "plan_output_dir" in plan_records
                else None
            )
            metrics_path = write_plan_metrics_csv(
                args.artifact_dir,
                args=args,
                plan_output_dir=plan_output_dir,
                log_path=log_path,
            )
            if metrics_path is not None:
                execution_results[-1]["metrics_csv"] = str(metrics_path)
        update_execution_status(args.artifact_dir, execution_results)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
