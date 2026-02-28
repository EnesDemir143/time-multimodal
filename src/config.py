"""Merkezi konfigürasyon yöneticisi — config/config.yaml → typed Python nesneleri.

Kullanım:
    from src.config import get_config

    cfg = get_config()
    cfg.seed                          # 42
    cfg.paths.raw_csv                 # Path(".../data/raw/csv/patient_01.csv")
    cfg.columns.raw_features          # ["age", "sex", ...]
    cfg.embeddings.radiological_dim   # 768
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ── Proje kökü ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


# ══════════════════════════════════════════════════════════════════════════
# Typed Dataclasses
# ══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PathsConfig:
    """Proje dosya yolları (proje köküne göre resolve edilmiş absolute Path)."""

    raw_csv: Path
    raw_images: Path
    tabpfn_features: Path
    tabpfn_features_clean: Path
    xray_lmdb: Path
    splits_dir: Path
    embeddings_h5: Path
    embeddings_tabular: Path
    embeddings_radiological: Path


@dataclass(frozen=True)
class CVConfig:
    """Cross-validation ayarları."""

    n_folds: int


@dataclass(frozen=True)
class ColumnsConfig:
    """Kolon tanımları."""

    patient_id: str
    label_source: str
    label_positive: str
    raw_features: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EmbeddingsConfig:
    """Embedding boyutları."""

    radiological_dim: int
    tabular_dim: int


@dataclass(frozen=True)
class ImageConfig:
    """Görüntü transform ayarları."""

    size: int
    imagenet_mean: tuple[float, ...]
    imagenet_std: tuple[float, ...]


@dataclass(frozen=True)
class LMDBConfig:
    """LMDB ayarları."""

    map_size_gb: int

    @property
    def map_size_bytes(self) -> int:
        """Map size in bytes."""
        return self.map_size_gb * 1024 * 1024 * 1024


@dataclass(frozen=True)
class Config:
    """Proje genelindeki tüm konfigürasyon."""

    seed: int
    cv: CVConfig
    paths: PathsConfig
    columns: ColumnsConfig
    embeddings: EmbeddingsConfig
    image: ImageConfig
    lmdb: LMDBConfig


# ══════════════════════════════════════════════════════════════════════════
# Loader
# ══════════════════════════════════════════════════════════════════════════


def _resolve_paths(raw: dict[str, str], root: Path) -> dict[str, Path]:
    """YAML'daki relative path string'lerini absolute Path nesnelerine çevirir."""
    return {key: root / value for key, value in raw.items()}


def load_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
    """YAML dosyasını oku ve typed Config nesnesine dönüştür.

    Args:
        config_path: YAML dosya yolu. Varsayılan: ``config/config.yaml``.

    Returns:
        Frozen ``Config`` dataclass instance.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        msg = f"Config dosyası bulunamadı: {config_path}"
        raise FileNotFoundError(msg)

    with open(config_path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    # ── Paths ────────────────────────────────────────────────────────────
    resolved = _resolve_paths(raw["paths"], PROJECT_ROOT)
    paths = PathsConfig(**resolved)

    # ── CV ────────────────────────────────────────────────────────────────
    cv = CVConfig(**raw["cv"])

    # ── Columns ──────────────────────────────────────────────────────────
    columns = ColumnsConfig(
        patient_id=raw["columns"]["patient_id"],
        label_source=raw["columns"]["label_source"],
        label_positive=raw["columns"]["label_positive"],
        raw_features=list(raw["columns"]["raw_features"]),
    )

    # ── Embeddings ───────────────────────────────────────────────────────
    embeddings = EmbeddingsConfig(**raw["embeddings"])

    # ── Image ────────────────────────────────────────────────────────────
    image = ImageConfig(
        size=raw["image"]["size"],
        imagenet_mean=tuple(raw["image"]["imagenet_mean"]),
        imagenet_std=tuple(raw["image"]["imagenet_std"]),
    )

    # ── LMDB ─────────────────────────────────────────────────────────────
    lmdb_cfg = LMDBConfig(**raw["lmdb"])

    return Config(
        seed=raw["seed"],
        cv=cv,
        paths=paths,
        columns=columns,
        embeddings=embeddings,
        image=image,
        lmdb=lmdb_cfg,
    )


# ══════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════

_CONFIG: Config | None = None


def get_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
    """Singleton erişim — config'i bir kez yükler, sonraki çağrılarda cache'den döner.

    Args:
        config_path: İlk çağrıda kullanılacak YAML yolu.

    Returns:
        Frozen ``Config`` instance.
    """
    global _CONFIG  # noqa: PLW0603
    if _CONFIG is None:
        _CONFIG = load_config(config_path)
    return _CONFIG
