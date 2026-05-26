from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

MANIFEST_NAMES = {
    "requirements.txt",
    "requirements-dev.txt",
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
    "environment.yml",
    "environment.yaml",
    "Pipfile",
}

DISALLOWED_DEPENDENCIES = {
    "torch",
    "pytorch",
    "numpy",
    "libero",
    "maniskill",
    "openvla",
    "transformers",
    "spikingjelly",
    "snntorch",
    "norse",
    "unitree",
}


def test_dependency_manifests_do_not_introduce_large_dependencies() -> None:
    manifests = [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.name in MANIFEST_NAMES
        and ".git" not in path.parts
    ]

    violations = []
    for manifest in manifests:
        text = manifest.read_text(encoding="utf-8").lower()
        for dependency in DISALLOWED_DEPENDENCIES:
            if dependency in text:
                violations.append(f"{manifest.relative_to(ROOT)}: {dependency}")

    assert violations == []


def test_project_contract_names_allowed_bootstrap_dependencies() -> None:
    contract = (ROOT / "docs" / "PROJECT_CONTRACT.md").read_text(encoding="utf-8")
    for dependency in ["Python stdlib", "Bash", "git", "pytest"]:
        assert dependency in contract


def test_installer_archives_are_not_present_or_tracked() -> None:
    forbidden_suffixes = {".zip", ":Zone.Identifier"}
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.iterdir()
        if any(path.name.endswith(suffix) for suffix in forbidden_suffixes)
    ]
    assert offenders == []
