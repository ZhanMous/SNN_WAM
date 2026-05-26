from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_PATHS = [
    "AGENTS.md",
    "README.md",
    "docs/project_plan.md",
    "docs/PROJECT_CONTRACT.md",
    "docs/EXPERIMENT_PROTOCOL.md",
    "docs/RESULT_ARTIFACTS.md",
    "docs/CLAIMS_LEDGER.md",
    "scripts/smoke_check.sh",
    ".agents/skills/snn-wam-project-guardrail/SKILL.md",
    "configs",
    "src",
    "results/runs",
    "results/tables",
    "results/figures",
]


def read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_required_repository_skeleton_exists() -> None:
    missing = [path for path in REQUIRED_PATHS if not (ROOT / path).exists()]
    assert missing == []


def test_stage_one_boundaries_are_documented() -> None:
    combined = "\n".join(
        read(path)
        for path in [
            "AGENTS.md",
            "README.md",
            "docs/PROJECT_CONTRACT.md",
            "docs/project_plan.md",
        ]
    )
    required_phrases = [
        "SNN temporal/world-action adapter",
        "不直接训练完整 VLA/WAM",
        "不直接上 OpenVLA 7B",
        "不直接上真实 Unitree",
        "不先写论文故事",
    ]
    for phrase in required_phrases:
        assert phrase in combined


def test_agents_required_workflow_and_quality_gates_are_documented() -> None:
    agents = read("AGENTS.md")
    required_phrases = [
        "Start with a plan",
        "Make the smallest coherent change",
        "config.yaml",
        "metrics.csv",
        "git_commit.txt",
        "docs/RESULT_ARTIFACTS.md",
        "Never silently squeeze, flatten, or reorder time dimensions",
        "[B, T, ...]",
    ]
    for phrase in required_phrases:
        assert phrase in agents


def test_claims_must_point_to_result_files() -> None:
    combined = "\n".join(
        read(path)
        for path in [
            "docs/PROJECT_CONTRACT.md",
            "docs/EXPERIMENT_PROTOCOL.md",
            "docs/RESULT_ARTIFACTS.md",
            "docs/CLAIMS_LEDGER.md",
        ]
    )
    required_phrases = [
        "每个 claim 必须",
        "Artifact ID",
        "Artifact IDs",
        "results/",
        "结果文件",
        "Evidence Files",
        "git_commit.txt",
        "environment.txt",
        "seeds.txt",
        "command.sh",
        "split.json",
    ]
    for phrase in required_phrases:
        assert phrase in combined


def test_canonical_scientific_guardrails_are_documented() -> None:
    combined = "\n".join(
        read(path)
        for path in [
            "docs/PROJECT_CONTRACT.md",
            "docs/EXPERIMENT_PROTOCOL.md",
        ]
    )
    required_phrases = [
        "Baseline Fairness",
        "相同数据 split",
        "相同 frozen encoders",
        "WAM Evidence Standard",
        "action-only baseline",
        "multi-step horizon degradation",
        "ES Evidence Standard",
        "surrogate-gradient",
        "Causal Data Rules",
        "future leakage",
        "不得包含 future observation",
        "reward",
        "success label",
    ]
    for phrase in required_phrases:
        assert phrase in combined


def test_result_artifact_registry_has_reproducibility_columns() -> None:
    registry = read("docs/RESULT_ARTIFACTS.md")
    required_phrases = [
        "| Artifact ID | Run ID | Result Files | Config | Commit | Environment | Seeds | Evaluation Split | Command | Notes |",
        "environment.txt",
        "seeds.txt",
        "command.sh",
        "split.json",
    ]
    for phrase in required_phrases:
        assert phrase in registry


def test_smoke_script_is_executable_bash() -> None:
    script = ROOT / "scripts" / "smoke_check.sh"
    assert script.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash")
    assert script.stat().st_mode & 0o111


def test_no_model_or_training_code_in_bootstrap() -> None:
    src_files = [
        path
        for path in (ROOT / "src").rglob("*")
        if path.is_file() and path.name != ".gitkeep"
    ]
    assert src_files == []
