from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "csv" / "patient_01.csv"
CLEAN_CSV = PROJECT_ROOT / "data" / "processed" / "tabpfn_features_clean.csv"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits" / "5fold"

# ── Config ───────────────────────────────────────────────────────────────
N_FOLDS = 5
SEED = 42


def create_labels(raw_path: Path, clean_path: Path) -> pd.DataFrame:
    """Raw CSV'den mortalite label oluştur ve clean CSV'deki hastalarla eşleştir.

    Label: destin_discharge == "Death" → 1, diğer tüm durumlar → 0
    """
    raw = pd.read_csv(raw_path, usecols=["patient_id", "destin_discharge"])
    clean = pd.read_csv(clean_path, usecols=["patient_id"])

    # Binary label: Death = 1, Survived (Home, Transfer, vb.) = 0
    raw["label"] = (raw["destin_discharge"] == "Death").astype(int)

    # Merge — sadece clean CSV'deki hastalar
    merged = clean.merge(raw[["patient_id", "label"]], on="patient_id", how="left")

    assert merged["label"].isna().sum() == 0, "Label eşleştirilemeyen hastalar var!"
    return merged


def create_splits(df: pd.DataFrame) -> None:
    """StratifiedGroupKFold ile 5-fold split oluştur ve CSV olarak kaydet.

    groups = patient_id → aynı hastanın tüm kayıtları aynı fold'da kalır.
    """
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    X = df["patient_id"].values
    y = df["label"].values
    groups = df["patient_id"].values  # hasta bazlı gruplama

    print(f"{'='*60}")
    print(f"  Stratified {N_FOLDS}-Fold CV Split — Mortalite")
    print(f"{'='*60}\n")
    print(f"  Toplam hasta : {len(df):,}")
    print(f"  Death   (1)  : {(y == 1).sum():,}")
    print(f"  Survived (0) : {(y == 0).sum():,}")
    print(f"  Oran (Survived/Death) : {(y == 0).sum() / (y == 1).sum():.2f}")
    print(f"  pos_weight önerisi    : {(y == 0).sum() / (y == 1).sum():.2f}")
    print(f"\n{'─'*60}\n")

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        train_df = df.iloc[train_idx][["patient_id", "label"]].reset_index(drop=True)
        val_df = df.iloc[val_idx][["patient_id", "label"]].reset_index(drop=True)

        train_path = SPLITS_DIR / f"fold_{fold_idx}_train.csv"
        val_path = SPLITS_DIR / f"fold_{fold_idx}_val.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)

        # İstatistikler
        t1 = (train_df["label"] == 1).sum()
        t0 = (train_df["label"] == 0).sum()
        v1 = (val_df["label"] == 1).sum()
        v0 = (val_df["label"] == 0).sum()

        print(f"  Fold {fold_idx}:")
        print(f"    Train → {len(train_df):,}  (Death={t1}, Survived={t0})")
        print(f"    Val   → {len(val_df):,}  (Death={v1}, Survived={v0})")
        print(f"    📁 {train_path.name}, {val_path.name}")
        print()

    print(f"{'─'*60}")
    print(f"  ✅ Tüm split'ler kaydedildi → {SPLITS_DIR}")
    print(f"{'='*60}")


def main() -> None:
    df = create_labels(RAW_CSV, CLEAN_CSV)
    create_splits(df)


if __name__ == "__main__":
    main()
