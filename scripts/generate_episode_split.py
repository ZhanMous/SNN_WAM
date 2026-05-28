#!/usr/bin/env python3
"""Generate episode-level split file from latent manifest.

This script reads a latent manifest JSON file, extracts unique episode IDs,
and generates a train/val/test split file with configurable ratios.

The split is episode-level only. No frame-level or window-level random split
is allowed for reportable experiments.

Usage:
    python scripts/generate_episode_split.py \
        --manifest latents/libero_spatial/dinov2_vits14/manifest.json \
        --output splits/libero_episode_split_seed20260528.json \
        --seed 20260528 \
        --train-ratio 0.8 \
        --val-ratio 0.1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to latent manifest JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for split file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260528,
        help="Random seed for shuffling. Default: 20260528.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of episodes for training. Default: 0.8.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of episodes for validation. Default: 0.1.",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="libero_spatial",
        help="LIBERO suite name. Default: libero_spatial.",
    )
    return parser.parse_args()


def load_manifest(manifest_path: Path) -> dict:
    """Load and validate latent manifest."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Validate required fields
    required_fields = ["encoder_id", "revision", "preprocessing_id", "latent_dim"]
    for field in required_fields:
        if field not in manifest:
            raise ValueError(f"Manifest missing required field: {field}")

    return manifest


def extract_episode_ids(manifest: dict) -> list[str]:
    """Extract unique episode IDs from manifest."""
    episode_ids = set()

    # Try to get from manifest metadata
    if "unique_episode_ids" in manifest:
        return sorted(manifest["unique_episode_ids"])

    # Fall back to extracting from demos
    for demo_info in manifest.get("demos", {}).values():
        if "episode_id" in demo_info:
            episode_ids.add(demo_info["episode_id"])

    if not episode_ids:
        raise ValueError("No episode IDs found in manifest")

    return sorted(episode_ids)


def split_episodes(
    episode_ids: list[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    """Split episodes into train/val/test sets."""
    # Validate ratios
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < 0:
        raise ValueError(
            f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) must be <= 1.0"
        )

    # Shuffle with seed
    rng = random.Random(seed)
    shuffled = episode_ids.copy()
    rng.shuffle(shuffled)

    # Calculate split indices
    n = len(shuffled)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))

    # Ensure we don't exceed total
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1

    n_test = n - n_train - n_val

    # Split
    train_ids = sorted(shuffled[:n_train])
    val_ids = sorted(shuffled[n_train:n_train + n_val])
    test_ids = sorted(shuffled[n_train + n_val:])

    # Validate no overlap
    assert len(set(train_ids) & set(val_ids)) == 0, "train/val overlap"
    assert len(set(train_ids) & set(test_ids)) == 0, "train/test overlap"
    assert len(set(val_ids) & set(test_ids)) == 0, "val/test overlap"

    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }


def compute_manifest_hash(manifest_path: Path) -> str:
    """Compute SHA256 hash of manifest file."""
    content = manifest_path.read_bytes()
    return hashlib.sha256(content).hexdigest()


def write_split_file(
    output_path: Path,
    splits: dict[str, list[str]],
    manifest: dict,
    manifest_path: Path,
    seed: int,
    suite: str,
) -> None:
    """Write split file with metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_hash = compute_manifest_hash(manifest_path)

    split_data = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "seed": seed,
            "split_unit": "episode",
            "benchmark": "LIBERO",
            "suite": suite,
            "source_manifest": str(manifest_path),
            "manifest_hash": manifest_hash,
            "encoder_id": manifest.get("encoder_id"),
            "revision": manifest.get("revision"),
            "preprocessing_id": manifest.get("preprocessing_id"),
            "latent_dim": manifest.get("latent_dim"),
            "total_episodes": sum(len(v) for v in splits.values()),
            "train_count": len(splits["train"]),
            "val_count": len(splits["val"]),
            "test_count": len(splits["test"]),
        },
        "splits": splits,
    }

    with open(output_path, "w") as f:
        json.dump(split_data, f, indent=2)


def main() -> int:
    args = parse_args()

    print(f"Manifest: {args.manifest}")
    print(f"Output: {args.output}")
    print(f"Seed: {args.seed}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Val ratio: {args.val_ratio}")
    print(f"Suite: {args.suite}")
    print()

    # Load manifest
    manifest = load_manifest(args.manifest)
    print(f"Loaded manifest: encoder={manifest['encoder_id']}, revision={manifest['revision']}")

    # Extract episode IDs
    episode_ids = extract_episode_ids(manifest)
    print(f"Found {len(episode_ids)} unique episodes")

    if len(episode_ids) < 3:
        print("ERROR: Need at least 3 episodes for train/val/test split")
        return 1

    # Split episodes
    splits = split_episodes(
        episode_ids,
        args.train_ratio,
        args.val_ratio,
        args.seed,
    )

    print(f"\nSplit results:")
    print(f"  Train: {len(splits['train'])} episodes")
    print(f"  Val:   {len(splits['val'])} episodes")
    print(f"  Test:  {len(splits['test'])} episodes")

    # Validate no overlap
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])
    assert len(train_set & val_set) == 0, "train/val overlap detected"
    assert len(train_set & test_set) == 0, "train/test overlap detected"
    assert len(val_set & test_set) == 0, "val/test overlap detected"
    print("  Overlap check: PASSED")

    # Write split file
    write_split_file(
        args.output,
        splits,
        manifest,
        args.manifest,
        args.seed,
        args.suite,
    )
    print(f"\nSplit file written to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
