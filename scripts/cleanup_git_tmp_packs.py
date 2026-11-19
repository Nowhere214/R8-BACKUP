#!/usr/bin/env python3
"""Remove stale Git tmp_pack files that can fill the disk on this project."""

from __future__ import annotations

import argparse
from pathlib import Path
import time


def fmt_bytes(size: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete stale .git/objects/pack/tmp_pack_* files."
    )
    parser.add_argument(
        "--repo",
        default=".",
        help="Repository root. Defaults to the current directory.",
    )
    parser.add_argument(
        "--min-age-seconds",
        type=int,
        default=60,
        help="Only delete tmp packs older than this age. Default: 60.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete all tmp packs regardless of age.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be deleted without removing files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).resolve()
    pack_dir = repo / ".git" / "objects" / "pack"

    if not pack_dir.is_dir():
        print(f"No Git pack directory found: {pack_dir}")
        return 0

    now = time.time()
    candidates = sorted(pack_dir.glob("tmp_pack_*"))
    stale: list[Path] = []
    skipped: list[Path] = []

    for path in candidates:
        age = now - path.stat().st_mtime
        if args.force or age >= args.min_age_seconds:
            stale.append(path)
        else:
            skipped.append(path)

    if not candidates:
        print("No tmp_pack files found.")
        return 0

    total = sum(path.stat().st_size for path in stale)
    action = "Would delete" if args.dry_run else "Deleting"
    print(f"{action} {len(stale)} tmp_pack files, total {fmt_bytes(total)}.")

    for path in stale:
        print(f"  {path} ({fmt_bytes(path.stat().st_size)})")
        if not args.dry_run:
            path.unlink()

    if skipped:
        skipped_size = sum(path.stat().st_size for path in skipped)
        print(
            f"Skipped {len(skipped)} recent tmp_pack files "
            f"({fmt_bytes(skipped_size)}). Use --force after stopping Git jobs."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
