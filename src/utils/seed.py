"""Deterministic seeding utilities."""

from __future__ import annotations

import os
import random
from typing import Any


def seed_everything(seed: int, *, deterministic: bool = True) -> dict[str, Any]:
    """Seed Python, NumPy, and PyTorch when available.

    Args:
        seed: Non-negative integer seed.
        deterministic: If true and PyTorch is available, request deterministic
            CuDNN behavior where supported.

    Returns:
        A small status dictionary naming which libraries were seeded.
    """

    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    status: dict[str, Any] = {"python": True, "numpy": False, "torch": False}

    try:
        import numpy as np  # type: ignore[import-not-found]

        np.random.seed(seed)
        status["numpy"] = True
    except ImportError:
        pass

    try:
        import torch  # type: ignore[import-not-found]

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        status["torch"] = True
    except ImportError:
        pass

    return status


__all__ = ["seed_everything"]
