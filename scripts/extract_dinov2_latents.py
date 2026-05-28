#!/usr/bin/env python3
"""Extract DINOv2 latents from LIBERO HDF5 demonstrations.

This script extracts frozen DINOv2 ViT-S/14 latents from LIBERO demonstration
files and stores them in HDF5 format with manifest and checksum for
reproducible experiments.

Usage:
    python scripts/extract_dinov2_latents.py \
        --dataset-root $LIBERO_DATASET_ROOT \
        --suite libero_spatial \
        --output-dir latents/libero_spatial/dinov2_vits14 \
        --model-id facebook/dinov2-small \
        --revision <pinned_commit_hash>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to LIBERO dataset root directory.",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="libero_spatial",
        help="LIBERO suite name.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "latents" / "libero_spatial" / "dinov2_vits14",
        help="Output directory for extracted latents.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="facebook/dinov2-small",
        help="HuggingFace model ID for DINOv2.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Pinned commit hash for DINOv2 model.",
    )
    parser.add_argument(
        "--output-token",
        choices=["cls", "mean"],
        default="cls",
        help="Which token to use as latent (cls or mean of patches).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for DINOv2 inference.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for DINOv2 inference.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of HDF5 files to process.",
    )
    parser.add_argument(
        --dry-run,
        action="store_true",
        help="Print what would be done without extracting latents.",
    )
    return parser.parse_args()


def find_hdf5_files(dataset_root: Path, suite: str) -> list[Path]:
    """Find all HDF5 files in the dataset root for the given suite."""
    suite_dir = dataset_root / suite
    if not suite_dir.exists():
        raise FileNotFoundError(f"Suite directory not found: {suite_dir}")

    hdf5_files = sorted(suite_dir.glob("*.hdf5")) + sorted(suite_dir.glob("*.h5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files found in {suite_dir}")

    return hdf5_files


def extract_latents_from_file(
    hdf5_path: Path,
    encoder: Any,
    processor: Any,
    device: str,
    batch_size: int,
    output_token: str,
) -> dict[str, Any]:
    """Extract latents from a single HDF5 file."""

    with h5py.File(hdf5_path, "r") as f:
        # Find demo groups
        demo_groups = []
        if "data" in f:
            data_group = f["data"]
            for key in sorted(data_group.keys()):
                if isinstance(data_group[key], h5py.Group):
                    demo_groups.append(("data/" + key, data_group[key]))
        else:
            # Try top-level groups
            for key in sorted(f.keys()):
                if isinstance(f[key], h5py.Group) and "obs" in f[key]:
                    demo_groups.append((key, f[key]))

        if not demo_groups:
            print(f"  Warning: No demo groups found in {hdf5_path}")
            return {}

        results = {}
        for demo_path, demo_group in demo_groups:
            if "obs" not in demo_group:
                print(f"  Warning: No obs group in {demo_path}")
                continue

            obs_group = demo_group["obs"]

            # Find image keys
            image_keys = []
            for key in obs_group.keys():
                if "rgb" in key or "image" in key or "agentview" in key or "eye_in_hand" in key:
                    image_keys.append(key)

            if not image_keys:
                print(f"  Warning: No image keys found in {demo_path}/obs")
                continue

            # Use first image key
            image_key = image_keys[0]
            images = obs_group[image_key]

            if len(images.shape) < 3:
                print(f"  Warning: Unexpected image shape {images.shape} in {demo_path}")
                continue

            # Extract latents
            T = images.shape[0]
            latents = []

            for start_idx in range(0, T, batch_size):
                end_idx = min(start_idx + batch_size, T)
                batch_images = images[start_idx:end_idx]

                # Convert to torch tensor
                if isinstance(batch_images, np.ndarray):
                    batch_tensor = torch.from_numpy(batch_images).float()
                else:
                    batch_tensor = torch.tensor(batch_images).float()

                # Normalize to [0, 1] if needed
                if batch_tensor.max() > 1.0:
                    batch_tensor = batch_tensor / 255.0

                # Move to device
                batch_tensor = batch_tensor.to(device)

                # Process through DINOv2
                inputs = processor(images=batch_tensor, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = encoder(**inputs)

                if output_token == "cls":
                    batch_latents = outputs.last_hidden_state[:, 0, :]
                else:
                    batch_latents = outputs.last_hidden_state[:, 1:, :].mean(dim=1)

                latents.append(batch_latents.cpu().half())

            latents = torch.cat(latents, dim=0)
            results[demo_path] = {
                "image_key": image_key,
                "latents": latents.numpy(),
                "shape": latents.shape,
                "dtype": "float16",
            }

        return results


def compute_checksum(data: np.ndarray) -> str:
    """Compute SHA256 checksum of numpy array."""
    return hashlib.sha256(data.tobytes()).hexdigest()


def save_latents(
    output_dir: Path,
    hdf5_path: Path,
    results: dict[str, Any],
    model_id: str,
    revision: str | None,
    output_token: str,
) -> Path:
    """Save extracted latents to HDF5 file with manifest and checksum."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create output filename based on input filename
    output_name = hdf5_path.stem + "_dinov2_vits14.hdf5"
    output_path = output_dir / output_name

    manifest = {
        "source_file": str(hdf5_path),
        "model_id": model_id,
        "revision": revision,
        "output_token": output_token,
        "latent_dim": 384,
        "dtype": "float16",
        "extraction_time": datetime.now(timezone.utc).isoformat(),
        "demos": {},
    }

    with h5py.File(output_path, "w") as f:
        # Create metadata group
        meta_group = f.create_group("_metadata")
        meta_group.attrs["model_id"] = model_id
        meta_group.attrs["revision"] = revision or "unknown"
        meta_group.attrs["output_token"] = output_token
        meta_group.attrs["latent_dim"] = 384
        meta_group.attrs["dtype"] = "float16"
        meta_group.attrs["source_file"] = str(hdf5_path)

        for demo_path, demo_data in results.items():
            # Create demo group
            demo_group = f.create_group(demo_path)

            # Save latents
            demo_group.create_dataset(
                "latents",
                data=demo_data["latents"],
                dtype="float16",
                compression="gzip",
                compression_opts=4,
            )

            # Save metadata
            demo_group.attrs["image_key"] = demo_data["image_key"]
            demo_group.attrs["shape"] = demo_data["shape"]
            demo_group.attrs["checksum"] = compute_checksum(demo_data["latents"])

            # Update manifest
            manifest["demos"][demo_path] = {
                "image_key": demo_data["image_key"],
                "shape": list(demo_data["shape"]),
                "checksum": demo_group.attrs["checksum"],
            }

    # Save manifest
    manifest_path = output_dir / f"{output_name}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return output_path


def main() -> int:
    args = parse_args()

    print(f"Dataset root: {args.dataset_root}")
    print(f"Suite: {args.suite}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model ID: {args.model_id}")
    print(f"Revision: {args.revision}")
    print(f"Output token: {args.output_token}")
    print(f"Device: {args.device}")
    print()

    # Find HDF5 files
    hdf5_files = find_hdf5_files(args.dataset_root, args.suite)
    if args.max_files:
        hdf5_files = hdf5_files[:args.max_files]

    print(f"Found {len(hdf5_files)} HDF5 files")

    if args.dry_run:
        print("\nDry run - would process:")
        for hdf5_path in hdf5_files:
            print(f"  {hdf5_path}")
        return 0

    # Load DINOv2 model
    print("\nLoading DINOv2 model...")
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        print("ERROR: transformers library is required. Install with: pip install transformers")
        return 1

    processor = AutoImageProcessor.from_pretrained(args.model_id, revision=args.revision)
    encoder = AutoModel.from_pretrained(args.model_id, revision=args.revision)
    encoder = encoder.to(args.device)
    encoder.eval()

    # Process each file
    total_demos = 0
    for i, hdf5_path in enumerate(hdf5_files, 1):
        print(f"\n[{i}/{len(hdf5_files)}] Processing {hdf5_path.name}...")

        results = extract_latents_from_file(
            hdf5_path,
            encoder,
            processor,
            args.device,
            args.batch_size,
            args.output_token,
        )

        if results:
            output_path = save_latents(
                args.output_dir,
                hdf5_path,
                results,
                args.model_id,
                args.revision,
                args.output_token,
            )
            total_demos += len(results)
            print(f"  Saved {len(results)} demos to {output_path}")
        else:
            print(f"  No latents extracted from {hdf5_path}")

    print(f"\nDone. Extracted latents from {total_demos} demos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
