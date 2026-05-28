from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_PATHS = [
    "AGENTS.md",
    "README.md",
    "docs/project_plan.md",
    "docs/PROJECT_CONTRACT.md",
    "docs/EXPERIMENT_PROTOCOL.md",
    "docs/ENVIRONMENT.md",
    "docs/DATA_CONTRACT.md",
    "docs/AUDIT_DATASET_LEAKAGE.md",
    "docs/LIBERO_ACTION_SEMANTICS.md",
    "docs/SPLIT_POLICY.md",
    "docs/NORMALIZATION_POLICY.md",
    "docs/FUTURE_LATENT_CONTRACT.md",
    "docs/LIBERO_DATA_CONTRACT.md",
    "docs/LIBERO_BOOTSTRAP.md",
    "docs/DATA_RISKS.md",
    "docs/LOCAL_PATHS_TEMPLATE.md",
    "docs/RESULT_ARTIFACTS.md",
    "docs/CLAIMS_LEDGER.md",
    "docs/WAM_GRU_ABLATION_REPORT_TEMPLATE.md",
    "configs/libero_spatial_mlp.yaml",
    "configs/libero_spatial_gru.yaml",
    "configs/libero_spatial_snn_lif.yaml",
    "configs/smoke/libero_spatial_action_only_smoke.yaml",
    "configs/smoke/libero_spatial_wam_gru.yaml",
    "configs/smoke/libero_spatial_gru_no_future.yaml",
    "scripts/smoke_check.sh",
    "scripts/quality_gate.sh",
    "scripts/check_environment.py",
    "scripts/check_result_artifacts.py",
    "scripts/smoke_libero_import.py",
    "scripts/smoke_libero_env_step.py",
    "scripts/inspect_libero_data.py",
    "scripts/inspect_libero_demo.py",
    "scripts/check_libero_action_alignment.py",
    "scripts/bootstrap_libero_check.py",
    "scripts/download_libero_minimal.sh",
    "environment.yml",
    "src/data/trajectory_window.py",
    "src/utils/config.py",
    "src/utils/seed.py",
    "src/utils/experiment_io.py",
    "src/train/eval_offline.py",
    ".agents/skills/snn-wam-project-guardrail/SKILL.md",
    "configs",
    "src",
    "results/runs",
    "results/tables",
    "results/figures",
    "results/smoke",
    "results/inspections",
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
        "command.txt",
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


def test_g0_to_g9_gate_ladder_is_documented() -> None:
    combined = "\n".join(
        read(path)
        for path in [
            "README.md",
            "docs/PROJECT_CONTRACT.md",
            "docs/EXPERIMENT_PROTOCOL.md",
        ]
    )
    required_phrases = [
        "G0",
        "Repo Gate",
        "G1",
        "Environment Gate",
        "torch/LIBERO import",
        "G1.5",
        "LIBERO Bootstrap Gate",
        "G2",
        "Dataset Gate",
        "no-future-leakage",
        "G3",
        "Model Gate",
        "forward shape",
        "G4",
        "Training Gate",
        "tiny-batch overfit",
        "G5",
        "Metric Gate",
        "future cosine",
        "G6",
        "Rollout Gate",
        "fixed initial states",
        "G7",
        "Robustness Gate",
        "frame drop",
        "G8",
        "Evidence Gate",
        "RESULT_ARTIFACTS.md",
        "G9",
        "Claim Gate",
        "every paper/report sentence points to evidence",
    ]
    for phrase in required_phrases:
        assert phrase in combined


def test_environment_gate_workflow_is_documented() -> None:
    combined = "\n".join(
        read(path)
        for path in [
            "README.md",
            "docs/ENVIRONMENT.md",
            "docs/PROJECT_CONTRACT.md",
        ]
    )
    required_phrases = [
        "snnwam-libero",
        "snnwam-maniskill",
        "torch/LIBERO import",
        "python scripts/check_environment.py",
        "--require torch --require libero",
        "Do not add ManiSkill, OpenVLA, or Unitree dependencies",
    ]
    for phrase in required_phrases:
        assert phrase in combined


def test_libero_smoke_workflow_is_documented() -> None:
    environment_doc = read("docs/ENVIRONMENT.md")
    required_phrases = [
        "scripts/smoke_libero_import.py",
        "scripts/smoke_libero_env_step.py",
        "--run-step",
        "OffScreenRenderEnv",
        "results/smoke/",
        "LIBERO_DATA_ROOT",
        "MUJOCO_GL",
        "PYOPENGL_PLATFORM",
        "do not train models",
        "do not download datasets",
        "do not modify LIBERO source code",
    ]
    for phrase in required_phrases:
        assert phrase in environment_doc


def test_data_contract_documents_shapes_and_leakage_risks() -> None:
    data_contract = read("docs/DATA_CONTRACT.md")
    required_phrases = [
        "scripts/inspect_libero_data.py --mock",
        "results/inspections/",
        "[T, H, W, C]",
        "[T, action_dim]",
        "action_history",
        "[history_len, action_dim]",
        "target_actions",
        "[action_horizon, action_dim]",
        "target_future_images",
        "images[t+1:t+1+future_horizon]",
        "Future Leakage Risks",
        "`action_history` accidentally includes future target actions",
        "Do not assume real LIBERO action dimension",
        "Current Implementation Boundary",
        "no large visual",
    ]
    for phrase in required_phrases:
        assert phrase in data_contract


def test_libero_real_data_contract_blocks_dataset_until_observed() -> None:
    combined = "\n".join(
        read(path)
        for path in [
            "docs/LIBERO_BOOTSTRAP.md",
            "docs/LIBERO_DATA_CONTRACT.md",
            "docs/DATA_RISKS.md",
            "docs/ENVIRONMENT.md",
            "docs/LOCAL_PATHS_TEMPLATE.md",
        ]
    )
    required_phrases = [
        "Observed Schema",
        "not observed",
        "G1.5 LIBERO Bootstrap Gate",
        "LIBERO_REPO_ROOT",
        "LIBERO_DATASET_ROOT",
        "LIBERO_DATA_ROOT",
        "download_libero_datasets.py --datasets libero_spatial",
        "Real action dimension unknown",
        "Camera keys unknown",
        "Language keys unknown",
        "Split format unknown",
        "Action alignment unknown",
        "blocked by G1.5",
        "Future leakage",
        "dry-run WAM training may run only as a smoke test",
        "G2 TrajectoryWindowDataset v1 may begin only after G1.5 has inspected at least one real LIBERO HDF5 demonstration file",
        "scripts/inspect_libero_demo.py",
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


def test_quality_gate_script_is_executable_bash() -> None:
    script = ROOT / "scripts" / "quality_gate.sh"
    assert script.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash")
    assert script.stat().st_mode & 0o111


def test_libero_smoke_scripts_are_executable_python() -> None:
    for script_name in [
        "smoke_libero_import.py",
        "smoke_libero_env_step.py",
        "inspect_libero_data.py",
        "inspect_libero_demo.py",
        "check_libero_action_alignment.py",
        "bootstrap_libero_check.py",
        "check_result_artifacts.py",
    ]:
        script = ROOT / "scripts" / script_name
        assert script.read_text(encoding="utf-8").startswith("#!/usr/bin/env python3")
        assert script.stat().st_mode & 0o111


def test_libero_download_wrapper_is_executable_bash() -> None:
    script = ROOT / "scripts" / "download_libero_minimal.sh"
    assert script.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash")
    assert script.stat().st_mode & 0o111


def test_only_phase1_offline_model_training_code_is_present_for_model_gate() -> None:
    allowed_src_files = {
        "src/.gitkeep",
        "src/__init__.py",
        "src/data/__init__.py",
        "src/data/split_normalization.py",
        "src/data/trajectory_window.py",
        "src/models/encoders.py",
        "src/models/__init__.py",
        "src/models/heads.py",
        "src/models/registry.py",
        "src/models/temporal_gru.py",
        "src/models/temporal_mlp.py",
        "src/train/__init__.py",
        "src/train/eval_offline.py",
        "src/train/metrics.py",
        "src/train/train_offline.py",
        "src/utils/__init__.py",
        "src/utils/config.py",
        "src/utils/experiment_io.py",
        "src/utils/seed.py",
    }
    src_files = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    assert src_files <= allowed_src_files

    forbidden_tokens = {"snn"}
    forbidden_paths = [
        path for path in src_files if any(token in path.lower() for token in forbidden_tokens)
    ]
    assert forbidden_paths == []
