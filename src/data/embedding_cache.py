from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ── Defaults ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_H5_PATH = PROJECT_ROOT / "data" / "embeddings" / "embeddings.h5"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits" / "5fold"

RADIOLOGICAL_DIM = 768
TABULAR_DIM = 192


# ══════════════════════════════════════════════════════════════════════════
# Writer
# ══════════════════════════════════════════════════════════════════════════
class EmbeddingCacheWriter:
    """HDF5 dosyasına embedding yazıcı.

    Args:
        h5_path: HDF5 dosya yolu. Yoksa oluşturulur.
    """

    def __init__(self, h5_path: Path | str = DEFAULT_H5_PATH) -> None:
        self.h5_path = Path(h5_path)
        self.h5_path.parent.mkdir(parents=True, exist_ok=True)
        self._file: h5py.File | None = None

    # ── Context manager ──────────────────────────────────────────────────

    def open(self) -> EmbeddingCacheWriter:
        """HDF5 dosyasını açar (append mode)."""
        self._file = h5py.File(self.h5_path, "a")

        # Grupları garanti et
        for grp in ("radiological", "tabular", "metadata"):
            if grp not in self._file:
                self._file.create_group(grp)

        return self

    def close(self) -> None:
        """HDF5 dosyasını kapatır."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> EmbeddingCacheWriter:
        return self.open()

    def __exit__(self, *_exc) -> None:
        self.close()

    # ── Yazma işlemleri ──────────────────────────────────────────────────

    def _ensure_open(self) -> h5py.File:
        if self._file is None:
            msg = "HDF5 dosyası açık değil. `open()` veya context manager kullanın."
            raise RuntimeError(msg)
        return self._file

    def save_radiological(
        self,
        patient_id: int,
        embedding: np.ndarray,
        xray_idx: int = 0,
        *,
        force: bool = False,
    ) -> None:
        """Radyolojik (RadJEPA) embedding'i kaydeder.

        Args:
            patient_id: Hasta ID.
            embedding: (768,) float32 numpy array (L2 normalize edilmiş).
            xray_idx: X-ray indeksi (bir hastanın birden fazla görüntüsü varsa).
            force: True ise varolan veriyi üzerine yazar.
        """
        f = self._ensure_open()
        embedding = np.asarray(embedding, dtype=np.float32)

        if embedding.shape != (RADIOLOGICAL_DIM,):
            msg = (
                f"Radyolojik embedding boyutu hatalı: beklenen ({RADIOLOGICAL_DIM},), "
                f"alınan {embedding.shape}"
            )
            raise ValueError(msg)

        key = f"p{patient_id}_{xray_idx}"
        grp = f["radiological"]

        if key in grp:
            if not force:
                return  # Zaten var, atla
            del grp[key]

        grp.create_dataset(key, data=embedding)

    def save_tabular(
        self,
        patient_id: int,
        embedding: np.ndarray,
        *,
        force: bool = False,
    ) -> None:
        """Tabular (TabPFN v2) embedding'i kaydeder.

        Args:
            patient_id: Hasta ID.
            embedding: (192,) float32 numpy array.
            force: True ise varolan veriyi üzerine yazar.
        """
        f = self._ensure_open()
        embedding = np.asarray(embedding, dtype=np.float32)

        if embedding.shape != (TABULAR_DIM,):
            msg = (
                f"Tabular embedding boyutu hatalı: beklenen ({TABULAR_DIM},), "
                f"alınan {embedding.shape}"
            )
            raise ValueError(msg)

        key = f"p{patient_id}"
        grp = f["tabular"]

        if key in grp:
            if not force:
                return  # Zaten var, atla
            del grp[key]

        grp.create_dataset(key, data=embedding)

    def save_metadata(
        self,
        patient_ids: np.ndarray,
        labels: np.ndarray,
        num_xrays: np.ndarray,
    ) -> None:
        """Metadata bilgilerini toplu kaydeder.

        Args:
            patient_ids: (N,) int64 array.
            labels: (N,) int32 array  (0=Survived, 1=Death).
            num_xrays: (N,) int32 array — hasta başına X-ray sayısı.
        """
        f = self._ensure_open()
        grp = f["metadata"]

        for name, data, dtype in [
            ("patient_ids", patient_ids, np.int64),
            ("labels", labels, np.int32),
            ("num_xrays", num_xrays, np.int32),
        ]:
            arr = np.asarray(data, dtype=dtype)
            if name in grp:
                del grp[name]
            grp.create_dataset(name, data=arr)

        # Cache oluşturma zamanı
        grp.attrs["created_at"] = datetime.now(tz=timezone.utc).isoformat()
        grp.attrs["radiological_dim"] = RADIOLOGICAL_DIM
        grp.attrs["tabular_dim"] = TABULAR_DIM

    # ── Bilgi ────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"EmbeddingCacheWriter(h5_path='{self.h5_path}')"


# ══════════════════════════════════════════════════════════════════════════
# Dataset (okuyucu)
# ══════════════════════════════════════════════════════════════════════════
class EmbeddingCacheDataset(Dataset):
    """Cached embedding'lerden okuyan PyTorch Dataset.

    Her örnek = (radiological_768, tabular_192, label).
    Birden fazla X-ray → ayrı sample (MortalityDataset ile uyumlu).

    Args:
        h5_path: HDF5 dosya yolu.
        fold: Fold numarası (0-4).
        split: "train" veya "val".
        splits_dir: Split CSV dizini.
    """

    def __init__(
        self,
        h5_path: Path | str = DEFAULT_H5_PATH,
        fold: int = 0,
        split: Literal["train", "val"] = "train",
        *,
        splits_dir: Path | str = SPLITS_DIR,
    ) -> None:
        super().__init__()

        self.h5_path = Path(h5_path)
        self.fold = fold
        self.split = split

        # ── 1) Split CSV → patient_id, label ────────────────────────────
        split_csv = Path(splits_dir) / f"fold_{fold}_{split}.csv"
        if not split_csv.exists():
            msg = f"Split dosyası bulunamadı: {split_csv}"
            raise FileNotFoundError(msg)

        split_df = pd.read_csv(split_csv)
        self._patient_ids: set[int] = set(split_df["patient_id"].values)
        self._labels: dict[int, int] = dict(
            zip(split_df["patient_id"], split_df["label"], strict=True)
        )

        # ── 2) HDF5'ten sample index inşa et ────────────────────────────
        self._h5: h5py.File | None = None  # Lazy open (fork-safe)
        self._samples: list[tuple[int, str, str]] = []
        # (patient_id, rad_key, tab_key)

        with h5py.File(self.h5_path, "r") as f:
            rad_keys = set(f["radiological"].keys()) if "radiological" in f else set()
            tab_keys = set(f["tabular"].keys()) if "tabular" in f else set()

        # Radyolojik key'lerden sample oluştur
        for rk in sorted(rad_keys):
            # rk format: p{patient_id}_{xray_idx}
            pid = int(rk.split("_")[0].lstrip("p"))
            if pid in self._patient_ids:
                tab_key = f"p{pid}"
                self._samples.append((pid, rk, tab_key))

        # Deterministik sıralama
        self._samples.sort(key=lambda x: (x[0], x[1]))

        # Tabular cache (split içindeki)
        self._tabular_available = tab_keys

    # ── HDF5 lazy open (fork-safe for DataLoader workers) ────────────────

    def _open_h5(self) -> h5py.File:
        """Her worker process'te HDF5'i yeniden aç."""
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    # ── Dataset interface ────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Tek bir örnek döndür.

        Returns:
            radiological: (768,) float32 tensor (L2 normalized).
            tabular: (192,) float32 tensor.
            label: int (0=Survived, 1=Death).
        """
        pid, rad_key, tab_key = self._samples[index]
        f = self._open_h5()

        # Radiological embedding
        rad_emb = torch.from_numpy(f["radiological"][rad_key][:].astype(np.float32))

        # Tabular embedding
        if tab_key in self._tabular_available:
            tab_emb = torch.from_numpy(f["tabular"][tab_key][:].astype(np.float32))
        else:
            tab_emb = torch.zeros(TABULAR_DIM, dtype=torch.float32)

        label = self._labels[pid]

        return rad_emb, tab_emb, label

    # ── Utility methods ──────────────────────────────────────────────────

    @property
    def num_samples(self) -> int:
        return len(self._samples)

    @property
    def class_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {0: 0, 1: 0}
        for pid, _, _ in self._samples:
            counts[self._labels[pid]] += 1
        return counts

    @property
    def pos_weight(self) -> float:
        counts = self.class_counts
        if counts[1] == 0:
            return 1.0
        return counts[0] / counts[1]

    def __repr__(self) -> str:
        counts = self.class_counts
        return (
            f"EmbeddingCacheDataset(fold={self.fold}, split='{self.split}', "
            f"samples={len(self)}, "
            f"survived={counts[0]}, death={counts[1]})"
        )

    def __del__(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None


# ══════════════════════════════════════════════════════════════════════════
# Helper fonksiyonlar
# ══════════════════════════════════════════════════════════════════════════
def is_cached(
    h5_path: Path | str,
    patient_id: int,
    modality: Literal["radiological", "tabular"],
) -> bool:
    """Belirtilen hasta embedding'inin cache'te olup olmadığını kontrol eder."""
    h5_path = Path(h5_path)
    if not h5_path.exists():
        return False

    with h5py.File(h5_path, "r") as f:
        if modality not in f:
            return False
        prefix = f"p{patient_id}"
        return any(k.startswith(prefix) for k in f[modality].keys())


def cache_stats(h5_path: Path | str = DEFAULT_H5_PATH) -> dict:
    """Cache istatistiklerini döndürür."""
    h5_path = Path(h5_path)
    if not h5_path.exists():
        return {"exists": False}

    with h5py.File(h5_path, "r") as f:
        n_rad = len(f["radiological"]) if "radiological" in f else 0
        n_tab = len(f["tabular"]) if "tabular" in f else 0
        attrs = dict(f["metadata"].attrs) if "metadata" in f else {}

        # Unique patient count from radiological keys
        rad_pids = set()
        if "radiological" in f:
            for k in f["radiological"].keys():
                pid = int(k.split("_")[0].lstrip("p"))
                rad_pids.add(pid)

    return {
        "exists": True,
        "h5_path": str(h5_path),
        "num_radiological_embeddings": n_rad,
        "num_tabular_embeddings": n_tab,
        "num_unique_patients_radiological": len(rad_pids),
        "created_at": attrs.get("created_at", "unknown"),
        "radiological_dim": attrs.get("radiological_dim", RADIOLOGICAL_DIM),
        "tabular_dim": attrs.get("tabular_dim", TABULAR_DIM),
    }
