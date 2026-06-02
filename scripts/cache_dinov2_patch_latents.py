#!/usr/bin/env python3
"""Cache DINOv2 spatial patch latents from LIBERO HDF5 demonstrations.

This script extracts frozen DINOv2 patch token embeddings from LIBERO
demonstration files and stores them in .pt format with metadata for
reproducible spatial patch-latent experiments.

Stored per-demo:
- patch_latents: [T, N, D] float16 (spatial patch tokens)
- actions: [T, A] float32 (preserved for alignment)
- timesteps: [T] int32

Metadata per-file:
- PatchLatentMetadata as JSON sidecar
- manifest.json with per-demo records

Usage:
    python scripts/cache_dinov2_patch_latents.py \
        --dataset-root $LIBERO_DATASET_ROOT \
        --suite libero_spatial \
        --output-dir latents/libero_spatial/dinov2_vits14_patch \
        --max-demos 1 \
        --max-frames 8 \
        --batch-size 2
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.encoders import DEFAULT_DINOV2_REVISION, PatchLatentMetadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to LIBERO dataset root directory.",
    )
    parser.add_argument("--suite", type=str, default="libero_spatial")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "latents" / "libero_spatial" / "dinov2_vits14_patch",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="facebook/dinov2-small",
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-demos", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def find_hdf5_files(dataset_root: Path, suite: str) -> list[Path]:
    candidate_dirs = [
        dataset_root / suite,
        dataset_root / "datasets" / suite,
        dataset_root / f"{suite}_no_noops",
    ]
    for d in candidate_dirs:
        if d.exists():
            files = sorted(d.glob("*.hdf5")) + sorted(d.glob("*.h5"))
            if files:
                return files
    raise FileNotFoundError(
        f"No HDF5 files for suite={suite!r} under {dataset_root}"
    )


def extract_patch_latents_from_file(
    hdf5_path: Path,
    encoder: Any,
    processor: Any,
    device: str,
    batch_size: int,
    image_size: int,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Extract patch latents and metadata from a single HDF5 file."""
    import h5py

    results: dict[str, Any] = {}
    with h5py.File(hdf5_path, "r") as f:
        demo_groups = []
        if "data" in f:
            for key in sorted(f["data"].keys()):
                if isinstance(f["data"][key], h5py.Group):
                    demo_groups.append((f"data/{key}", f["data"][key]))

        for demo_path, demo_group in demo_groups:
            if "obs" not in demo_group:
                continue
            obs = demo_group["obs"]
            image_keys = [
                k
                for k in obs.keys()
                if any(t in k for t in ("rgb", "image", "agentview", "eye_in_hand"))
            ]
            if not image_keys:
                continue

            image_key = image_keys[0]
            images = obs[image_key]
            if len(images.shape) < 3:
                continue

            T = images.shape[0]
            if max_frames is not None:
                T = min(T, max_frames)

            actions = None
            if "actions" in demo_group:
                actions = np.array(demo_group["actions"][:T], dtype=np.float32)

            patch_latents_list: list[torch.Tensor] = []
            for start in range(0, T, batch_size):
                end = min(start + batch_size, T)
                batch_np = np.array(images[start:end])
                batch_tensor = torch.from_numpy(batch_np).float()
                if batch_tensor.max() > 1.0:
                    batch_tensor = batch_tensor / 255.0
                batch_tensor = batch_tensor.to(device)

                # Resize to model input size if needed
                h, w = batch_tensor.shape[1], batch_tensor.shape[2]
                if h != image_size or w != image_size:
                    batch_tensor = torch.nn.functional.interpolate(
                        batch_tensor.permute(0, 3, 1, 2),
                        size=(image_size, image_size),
                        mode="bilinear",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)

                pil_images = []
                from torchvision.transforms.functional import to_pil_image

                for img in batch_tensor:
                    # to_pil_image expects [C, H, W] format
                    if img.ndim == 3 and img.shape[-1] == 3:
                        img = img.permute(2, 0, 1)
                    pil_images.append(to_pil_image(img.cpu()))
                inputs = processor(images=pil_images, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    if hasattr(encoder, "forward_features"):
                        out = encoder.forward_features(inputs["pixel_values"])
                        if isinstance(out, dict) and "patch_tokens" in out:
                            patches = out["patch_tokens"]
                        else:
                            patches = out.last_hidden_state[:, 1:, :]
                    else:
                        out = encoder(inputs["pixel_values"])
                        patches = out.last_hidden_state[:, 1:, :]

                patch_latents_list.append(patches.cpu().half())

            patch_latents = torch.cat(patch_latents_list, dim=0)[:T]

            episode_id = f"{hdf5_path.stem}:{demo_path.split('/')[-1]}"
            results[demo_path] = {
                "episode_id": episode_id,
                "patch_latents": patch_latents,
                "actions": actions,
                "timesteps": np.arange(T, dtype=np.int32),
                "image_key": image_key,
                "shape": list(patch_latents.shape),
            }

    return results


def main() -> int:
    args = parse_args()
    revision = args.revision or DEFAULT_DINOV2_REVISION

    print(f"Dataset root: {args.dataset_root}")
    print(f"Suite: {args.suite}")
    print(f"Output dir: {args.output_dir}")
    print(f"Model: {args.model_id}")
    print(f"Revision: {revision}")
    print(f"Image size: {args.image_size}, Patch size: {args.patch_size}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    if args.max_demos:
        print(f"Max demos: {args.max_demos}")
    if args.max_frames:
        print(f"Max frames per demo: {args.max_frames}")
    print()

    hdf5_files = find_hdf5_files(args.dataset_root, args.suite)
    print(f"Found {len(hdf5_files)} HDF5 files")

    if args.dry_run:
        print("\nDry run - would process:")
        for p in hdf5_files:
            print(f"  {p}")
        return 0

    from transformers import AutoImageProcessor, AutoModel

    print("\nLoading DINOv2 model...")
    processor = AutoImageProcessor.from_pretrained(args.model_id, revision=revision)
    model = AutoModel.from_pretrained(args.model_id, revision=revision)
    model = model.to(args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    num_patches = (args.image_size // args.patch_size) ** 2
    feature_dim = model.config.hidden_size

    metadata = PatchLatentMetadata(
        encoder_name=args.model_id.split("/")[-1],
        encoder_type="dinov2_patch",
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_patches=num_patches,
        feature_dim=feature_dim,
        include_cls=False,
        dtype="float16",
        normalization="dino_internal",
        source_dataset=f"{args.suite}",
        revision=revision,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata.to_dict(), indent=2)
    )

    total_demos = 0
    demo_count = 0
    for i, hdf5_path in enumerate(hdf5_files, 1):
        if args.max_demos is not None and demo_count >= args.max_demos:
            break

        print(f"\n[{i}/{len(hdf5_files)}] {hdf5_path.name}...")
        results = extract_patch_latents_from_file(
            hdf5_path,
            model,
            processor,
            args.device,
            args.batch_size,
            args.image_size,
            max_frames=args.max_frames,
        )

        if not results:
            print("  No demos extracted")
            continue

        out_name = hdf5_path.stem + "_dinov2_vits14_patch.pt"
        out_path = args.output_dir / out_name

        save_data: dict[str, Any] = {}
        manifest_demos: dict[str, Any] = {}
        for demo_path, demo_data in results.items():
            save_data[demo_path] = {
                "patch_latents": demo_data["patch_latents"],
                "actions": demo_data["actions"],
                "timesteps": demo_data["timesteps"],
            }
            manifest_demos[demo_path] = {
                "episode_id": demo_data["episode_id"],
                "image_key": demo_data["image_key"],
                "shape": demo_data["shape"],
            }

        torch.save(save_data, out_path)

        manifest = {
            "encoder": metadata.to_dict(),
            "source_file": str(hdf5_path),
            "output_file": str(out_path),
            "extraction_time": datetime.now(timezone.utc).isoformat(),
            "total_demos": len(results),
            "demos": manifest_demos,
        }
        manifest_path = args.output_dir / f"{out_name}_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        total_demos += len(results)
        demo_count += len(results)
        print(f"  Saved {len(results)} demos -> {out_path}")

    print(f"\nDone. Extracted {total_demos} demos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
