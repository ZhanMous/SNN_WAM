#!/usr/bin/env python3
"""Resume-capable downloader for LIBERO datasets.

Downloads individual HDF5 files from Hugging Face with wget -c (resume).
Can be run multiple times to pick up where it left off.

Usage:
    python scripts/download_libero_resume.py [--suite libero_spatial] [--dry-run]

Requires: wget (system), h5py (for verification)
"""
import argparse
import os
import subprocess
import sys
import time

HF_BASE = "https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main"

TASK_NAMES = {
    "libero_spatial": [
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
    ],
}

MAX_RETRIES = 5
RETRY_DELAY_BASE = 10  # seconds, exponential backoff


def get_dataset_root() -> str:
    """Return the directory that contains suite subdirectories (e.g. libero_spatial/).

    Checks LIBERO_DATASET_ROOT and LIBERO_DATA_ROOT.  Falls back to
    ~/data/libero/datasets which is where the official LIBERO downloader
    places files when LIBERO_DATASET_ROOT=~/data/libero.
    """
    for var in ("LIBERO_DATASET_ROOT", "LIBERO_DATA_ROOT"):
        val = os.environ.get(var)
        if val:
            # If the env var points to a dir that already contains suite dirs, use it.
            # Otherwise check for a datasets/ subdirectory (official LIBERO layout).
            if os.path.isdir(os.path.join(val, "libero_spatial")):
                return val
            datasets_sub = os.path.join(val, "datasets")
            if os.path.isdir(os.path.join(datasets_sub, "libero_spatial")):
                return datasets_sub
            return val
    return os.path.expanduser("~/data/libero/datasets")


def expected_filename(task_name: str) -> str:
    """Return the HDF5 filename expected for a given task name."""
    return f"{task_name}_demo.hdf5"


def url_for(suite: str, filename: str) -> str:
    """Return the Hugging Face download URL for a suite file."""
    return f"{HF_BASE}/{suite}/{filename}"


def file_is_valid_hdf5(path: str) -> bool:
    """Check if a file is a valid HDF5 file.

    Falls back to checking the HDF5 magic bytes if h5py is not installed.
    """
    try:
        import h5py
        with h5py.File(path, "r") as f:
            _ = list(f.keys())
        return True
    except ImportError:
        # h5py not available — check magic bytes (HDF5 signature)
        try:
            with open(path, "rb") as f:
                magic = f.read(8)
            # HDF5 magic: bytes 0x89 0x48 0x44 0x46 0x0d 0x0a 0x1a 0x0a
            return magic == b"\x89HDF\r\n\x1a\n"
        except OSError:
            return False
    except (OSError, ValueError):
        return False


def download_one(url: str, dest: str, dry_run: bool = False) -> bool:
    """Download a single file with wget -c (resume). Returns True on success."""
    if dry_run:
        print(f"  [dry-run] wget -c -O {dest} {url}")
        return True

    for attempt in range(1, MAX_RETRIES + 1):
        delay = RETRY_DELAY_BASE * (2 ** (attempt - 1))
        try:
            result = subprocess.run(
                [
                    "wget", "-c", "--timeout=60", "--tries=3",
                    "--retry-connrefused", "--wait=5",
                    "-O", dest, url,
                ],
                timeout=600,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return True
            print(f"  wget exit {result.returncode} (attempt {attempt}/{MAX_RETRIES})")
            if result.stderr:
                for line in result.stderr.strip().splitlines()[-3:]:
                    print(f"    {line}")
        except subprocess.TimeoutExpired:
            print(f"  wget timed out (attempt {attempt}/{MAX_RETRIES})")
        except Exception as e:
            print(f"  error: {e} (attempt {attempt}/{MAX_RETRIES})")

        if attempt < MAX_RETRIES:
            print(f"  retrying in {delay}s...")
            time.sleep(delay)

    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", default="libero_spatial", choices=list(TASK_NAMES.keys()))
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true", help="Only check existing files, don't download")
    parser.add_argument("--clean-cache", action="store_true",
                        help="Remove stale Hugging Face download cache before downloading")
    args = parser.parse_args()

    dataset_root = args.dataset_root or get_dataset_root()
    suite_dir = os.path.join(dataset_root, args.suite)
    tasks = TASK_NAMES[args.suite]

    print(f"Suite:        {args.suite}")
    print(f"Dataset root: {dataset_root}")
    print(f"Suite dir:    {suite_dir}")
    print(f"Expected:     {len(tasks)} HDF5 files")
    print()

    os.makedirs(suite_dir, exist_ok=True)

    # Clean stale HF download cache if requested
    if args.clean_cache:
        import shutil
        cache_dir = os.path.join(dataset_root, ".cache")
        if os.path.isdir(cache_dir):
            print(f"Removing stale HF cache: {cache_dir}")
            shutil.rmtree(cache_dir)
            print()

    existing = []
    missing = []
    invalid = []

    for task in tasks:
        fname = expected_filename(task)
        fpath = os.path.join(suite_dir, fname)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            if file_is_valid_hdf5(fpath):
                existing.append((fname, size_mb))
            else:
                invalid.append((fname, size_mb))
        else:
            missing.append(fname)

    print(f"Status: {len(existing)} ok, {len(invalid)} corrupt, {len(missing)} missing")
    if existing:
        print("  OK:")
        for fname, mb in existing:
            print(f"    {fname} ({mb:.0f} MB)")
    if invalid:
        print("  CORRUPT (will re-download):")
        for fname, mb in invalid:
            print(f"    {fname} ({mb:.0f} MB)")
    if missing:
        print("  MISSING:")
        for fname in missing:
            print(f"    {fname}")
    print()

    if args.verify_only:
        sys.exit(0 if not missing and not invalid else 1)

    to_download = missing + [f for f, _ in invalid]
    if not to_download:
        print("All files present and valid. Nothing to do.")
        return

    print(f"Downloading {len(to_download)} file(s)...")
    print()

    succeeded = 0
    failed = 0

    for i, fname in enumerate(to_download, 1):
        fpath = os.path.join(suite_dir, fname)
        url = url_for(args.suite, fname)
        print(f"[{i}/{len(to_download)}] {fname}")

        # Remove corrupt partial file before retry
        if fname in [f for f, _ in invalid] and os.path.isfile(fpath):
            os.remove(fpath)
            print(f"  removed corrupt file")

        ok = download_one(url, fpath, dry_run=args.dry_run)
        if ok and not args.dry_run:
            if file_is_valid_hdf5(fpath):
                size_mb = os.path.getsize(fpath) / (1024 * 1024)
                print(f"  OK ({size_mb:.0f} MB)")
                succeeded += 1
            else:
                print(f"  WARN: downloaded but HDF5 validation failed")
                failed += 1
        elif ok:
            succeeded += 1
        else:
            print(f"  FAILED after {MAX_RETRIES} retries")
            failed += 1

    print()
    print(f"Done: {succeeded} succeeded, {failed} failed")
    if failed:
        print("Re-run this script to retry failed downloads (wget -c will resume).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
