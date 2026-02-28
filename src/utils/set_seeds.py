import os
import random

import numpy as np
import torch

from src.config import get_config


def set_seeds(seed: int | None = None) -> torch.Generator:
    """Set all random seeds for deterministic reproducibility and return a torch Generator.

    Seed varsayılan olarak ``config/config.yaml``'dan okunur.

    Seeds set:
        - Python stdlib ``random``
        - NumPy
        - PyTorch (CPU + MPS)
        - Environment variables for single-thread determinism

    Args:
        seed: Seed değeri. ``None`` ise config'den okunur.

    Returns:
        torch.Generator: MPS (varsa) veya CPU üzerinde seeded generator.
    """
    if seed is None:
        seed = get_config().seed
    # ── Python & NumPy ──────────────────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)

    # ── PyTorch CPU ─────────────────────────────────────────────────
    torch.manual_seed(seed)

    # ── PyTorch MPS (Apple Silicon) ─────────────────────────────────
    # torch.manual_seed already seeds MPS internally since PyTorch 2.x,
    # but we configure deterministic behaviour flags explicitly.
    if torch.backends.mps.is_available():
        # MPS does not have a separate seed API; manual_seed covers it.
        # Force deterministic ops where possible.
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # disable caching allocator non-determinism

    # ── PyTorch CUDA (for DGX / future GPU runs) ───────────────────
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # ── Deterministic flags ─────────────────────────────────────────
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ── Single-thread determinism (pipeline FAZ 1.1) ────────────────
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # ── Reusable Generator ──────────────────────────────────────────
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    return generator
