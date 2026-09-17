"""Shared reproducibility and validation helpers for fair experiments."""

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

__all__ = [
    "set_seed",
    "file_sha256",
    "validate_weight_files",
    "count_images",
    "assert_finite",
    "build_run_metadata",
    "write_run_metadata",
]


def set_seed(seed: int) -> None:
    """Seed the random number generators used by the experiment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_weight_files(weights: Mapping[str, Path]) -> None:
    """Raise if any named weight path is not a regular file."""
    missing = []
    for name, value in weights.items():
        path = Path(value)
        if not path.is_file():
            missing.append(f"{name}={path}")
    if missing:
        raise FileNotFoundError("Missing weight files: " + ", ".join(missing))


def count_images(root: Path) -> int:
    """Count supported image files directly under *root*."""
    return sum(
        1
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def assert_finite(name: str, *values: torch.Tensor) -> None:
    """Raise when any supplied tensor contains NaN or infinite values."""
    for value in values:
        if not torch.isfinite(value).all().item():
            raise FloatingPointError(f"{name} contains NaN or Inf")


def build_run_metadata(
    args,
    device: torch.device,
    weights: Mapping[str, Path],
    content_dir: Path,
    style_dir: Path,
) -> dict:
    """Collect reproducibility metadata for a run."""
    cuda_name = None
    if device.type == "cuda" and torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(device)

    return {
        "argv": sys.argv.copy(),
        "args": vars(args).copy(),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_name": cuda_name,
        "content_dir": str(content_dir),
        "content_count": count_images(content_dir),
        "style_dir": str(style_dir),
        "style_count": count_images(style_dir),
        "weights": {
            name: {"path": str(path), "sha256": file_sha256(Path(path))}
            for name, path in weights.items()
        },
    }


def write_run_metadata(path: Path, metadata: dict) -> None:
    """Write run metadata as indented UTF-8 JSON."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
