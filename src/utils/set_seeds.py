import os
import random

import numpy as np
import torch


def set_seeds(seed: int = 42) -> torch.Generator:
    """Set all random seeds for deterministic reproducibility and return a torch Generator.

    Seeds set:
        - Python stdlib `random`
        - NumPy
        - PyTorch (CPU + MPS)
        - Environment variables for single-thread determinism

    Args:
        seed: The seed value to use everywhere. Defaults to 42.

    Returns:
        torch.Generator: A manually seeded generator (on MPS if available, else CPU).
                         Pass this to DataLoader, random ops, etc.
    """
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
