"""Multimodal Mortality Dataset — LMDB (X-ray) + Tabular (CSV).

Her örnek = (image_tensor, tabular_tensor, label) üçlüsüdür.
Bir hastanın birden fazla X-ray görüntüsü varsa, her biri ayrı örnek olur.

Kullanım:
    from src.data.dataset import MortalityDataset

    ds = MortalityDataset(fold=0, split="train")
    img, tab, label = ds[0]
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Literal

import lmdb
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ── Defaults ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

LMDB_PATH = PROJECT_ROOT / "data" / "processed" / "xray.lmdb"
TABULAR_CSV = PROJECT_ROOT / "data" / "processed" / "tabpfn_features_clean.csv"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits" / "5fold"

# ImageNet normalization (RadJEPA pretrained)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Feature columns (patient_id hariç tüm sütunlar)
FEATURE_COLUMNS: list[str] | None = None  # Lazy-loaded at runtime


def _default_transform(image_size: int = 224) -> transforms.Compose:
    """Deterministic inference transform — no augmentation."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class MortalityDataset(Dataset):
    """PyTorch Dataset — her örnek bir (image, tabular, label) üçlüsüdür.

    Bir hastanın birden fazla X-ray'i varsa her biri ayrı index olarak döner.
    Tabular features tüm görüntüler için aynıdır (hasta-seviyesi).

    Args:
        fold: Fold numarası (0-4).
        split: "train" veya "val".
        lmdb_path: LMDB veritabanı yolu.
        tabular_csv: Temiz tabular CSV yolu.
        splits_dir: Split dosyalarının bulunduğu dizin.
        transform: Görüntü transform pipeline'ı. None → varsayılan (224×224 + ImageNet norm).
        image_size: Varsayılan transform için görüntü boyutu.
    """

    def __init__(
        self,
        fold: int = 0,
        split: Literal["train", "val"] = "train",
        *,
        lmdb_path: Path | str = LMDB_PATH,
        tabular_csv: Path | str = TABULAR_CSV,
        splits_dir: Path | str = SPLITS_DIR,
        transform: transforms.Compose | None = None,
        image_size: int = 224,
    ) -> None:
        super().__init__()

        self.fold = fold
        self.split = split
        self.lmdb_path = Path(lmdb_path)
        self.transform = transform or _default_transform(image_size)

        # ── 1) Split CSV oku → patient_id, label ────────────────────────
        split_csv = Path(splits_dir) / f"fold_{fold}_{split}.csv"
        if not split_csv.exists():
            msg = f"Split dosyası bulunamadı: {split_csv}"
            raise FileNotFoundError(msg)

        split_df = pd.read_csv(split_csv)
        self._patient_ids: set[int] = set(split_df["patient_id"].values)
        self._labels: dict[int, int] = dict(
            zip(split_df["patient_id"], split_df["label"], strict=True)
        )

        # ── 2) Tabular features yükle ───────────────────────────────────
        tab_df = pd.read_csv(Path(tabular_csv))
        feature_cols = [c for c in tab_df.columns if c != "patient_id"]
        self.feature_columns = feature_cols

        # patient_id → feature tensor (float32)
        self._tabular: dict[int, torch.Tensor] = {}
        for _, row in tab_df.iterrows():
            pid = int(row["patient_id"])
            if pid in self._patient_ids:
                feats = row[feature_cols].values.astype(np.float32)
                # NaN → 0.0 (TabPFN embedding aşamasında NaN korunur,
                # ama ham tensor için güvenli default)
                feats = np.nan_to_num(feats, nan=0.0)
                self._tabular[pid] = torch.from_numpy(feats)

        # ── 3) LMDB key → (patient_id, lmdb_key) eşleştirmesi ──────────
        self._env: lmdb.Environment | None = None  # lazy open
        self._samples: list[tuple[int, str]] = []  # (patient_id, lmdb_key)

        env = lmdb.open(str(self.lmdb_path), readonly=True, lock=False, subdir=True)
        with env.begin() as txn:
            all_keys: list[str] = json.loads(txn.get(b"__keys__").decode())
        env.close()

        for key in all_keys:
            # key format: pXXXXXXXX/sXXXXXXXX/XXXXXXXX_XXXX
            pid_str = key.split("/")[0]  # "p10030053"
            pid = int(pid_str.lstrip("p"))  # 10030053

            if pid in self._patient_ids:
                self._samples.append((pid, key))

        # Deterministik sıralama
        self._samples.sort(key=lambda x: (x[0], x[1]))

        # ── 4) İstatistikler ────────────────────────────────────────────
        patients_with_images = {s[0] for s in self._samples}
        self._patients_no_image = self._patient_ids - patients_with_images

    # ── LMDB lazy open (fork-safe for DataLoader workers) ────────────────

    def _open_lmdb(self) -> lmdb.Environment:
        """Her worker process'te LMDB'yi yeniden aç (fork-safe)."""
        if self._env is None:
            self._env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                subdir=True,
                readahead=False,  # random access için readahead kapalı
                meminit=False,
            )
        return self._env

    # ── Dataset interface ────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Tek bir örnek döndür.

        Returns:
            image: (3, H, W) float32 tensor (normalized).
            tabular: (n_features,) float32 tensor.
            label: int (0 = Survived, 1 = Death).
        """
        pid, lmdb_key = self._samples[index]

        # ── Image ────────────────────────────────────────────────────────
        env = self._open_lmdb()
        with env.begin() as txn:
            raw_bytes = txn.get(lmdb_key.encode("utf-8"))

        if raw_bytes is None:
            msg = f"LMDB'de key bulunamadı: {lmdb_key}"
            raise KeyError(msg)

        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        image_tensor: torch.Tensor = self.transform(image)

        # ── Tabular ─────────────────────────────────────────────────────
        tabular_tensor = self._tabular.get(
            pid,
            torch.zeros(len(self.feature_columns), dtype=torch.float32),
        )

        # ── Label ────────────────────────────────────────────────────────
        label = self._labels[pid]

        return image_tensor, tabular_tensor, label

    # ── Utility methods ──────────────────────────────────────────────────

    @property
    def num_features(self) -> int:
        """Tabular feature sayısı."""
        return len(self.feature_columns)

    @property
    def num_samples(self) -> int:
        """Toplam örnek sayısı (image-level)."""
        return len(self._samples)

    @property
    def class_counts(self) -> dict[int, int]:
        """Label başına örnek sayısı."""
        counts: dict[int, int] = {0: 0, 1: 0}
        for pid, _ in self._samples:
            counts[self._labels[pid]] += 1
        return counts

    @property
    def pos_weight(self) -> float:
        """BCEWithLogitsLoss için önerilen pos_weight (survived / death)."""
        counts = self.class_counts
        if counts[1] == 0:
            return 1.0
        return counts[0] / counts[1]

    def __repr__(self) -> str:
        counts = self.class_counts
        return (
            f"MortalityDataset(fold={self.fold}, split='{self.split}', "
            f"samples={len(self)}, "
            f"survived={counts[0]}, death={counts[1]}, "
            f"features={self.num_features}, "
            f"patients_no_image={len(self._patients_no_image)})"
        )

    def __del__(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
