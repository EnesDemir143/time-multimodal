"""
TabPFN için özellik çıkarımı + multimodal filtreleme.

Pipeline:
    patient_01.csv → clean → feature eng → drop high-NaN cols
    → LMDB intersection (sadece X-ray'li hastalar) → tabpfn_features_clean.csv

Kullanım:
    uv run python scripts/extract_tabpfn_features.py
"""
from __future__ import annotations

import json
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd

from src.config import get_config

# ── Config ───────────────────────────────────────────────────────────────
_cfg = get_config()

RAW_CSV = _cfg.paths.raw_csv
OUT_RAW = _cfg.paths.tabpfn_features          # ara çıktı (tüm hastalar)
OUT_CLEAN = _cfg.paths.tabpfn_features_clean  # nihai çıktı (multimodal)
LMDB_PATH = _cfg.paths.xray_lmdb

# NaN oranı bu eşiğin üzerindeki kolonlar düşürülür
NAN_THRESHOLD = 0.50

# ── Kullanılacak ham kolonlar (config'den) ───────────────────────────────
RAW_COLS = [_cfg.columns.patient_id] + list(_cfg.columns.raw_features)


def load_and_select(path: Path) -> pd.DataFrame:
    """CSV'yi yükle ve sadece gerekli kolonları seç."""
    df = pd.read_csv(path)
    missing = set(RAW_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV'de şu kolonlar yok: {missing}")
    return df[RAW_COLS].copy()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Eksik / imkansız değerleri NaN'e çevir."""

    # age == 0  →  NaN  (newborn değilse missing demektir)
    df["age"] = df["age"].replace(0, np.nan)

    # sex  →  binary encode  (MALE=1, FEMALE=0)
    df["sex"] = df["sex"].map({"MALE": 1, "FEMALE": 0})

    # 0 değerleri fizyolojik olarak imkansız → NaN
    zero_to_nan_cols = [
        "bp_max_first_emerg",
        "bp_min_first_emerg",
        "glu_first_emerg",
        "bp_max_last_emerg",
        "bp_min_last_emerg",
    ]
    for col in zero_to_nan_cols:
        df[col] = df[col].replace(0, np.nan)

    # temp 0.0 → NaN  (vücut sıcaklığı 0 olamaz)
    df["temp_first_emerg"] = df["temp_first_emerg"].replace(0.0, np.nan)
    df["temp_last_emerg"] = df["temp_last_emerg"].replace(0.0, np.nan)

    return df


def drop_high_nan_columns(df: pd.DataFrame) -> pd.DataFrame:
    """NaN oranı %50'nin üzerindeki kolonları düşür."""
    nan_ratios = df.drop(columns=["patient_id"]).isna().mean()
    drop_cols = nan_ratios[nan_ratios > NAN_THRESHOLD].index.tolist()

    if drop_cols:
        print(f"  🗑️  >{NAN_THRESHOLD:.0%} NaN — düşürülen kolonlar: {drop_cols}")
        df = df.drop(columns=drop_cols)
    else:
        print(f"  ✅ >{NAN_THRESHOLD:.0%} NaN kolon yok")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hesaplanmış özellikleri ekle."""

    # Ateş değişimi (düşüş pozitif)
    df["temp_drop"] = df["temp_first_emerg"] - df["temp_last_emerg"]

    # SpO2 düşüşü (düşüş pozitif — kritik!)
    df["sat_drop"] = df["sat_02_first_emerg"] - df["sat_02_last_emerg"]

    # Shock Index = HR / Sistolik BP  (bp_max NaN ise sonuç da NaN olur)
    df["shock_index"] = df["hr_first_emerg"] / df["bp_max_first_emerg"]

    # Nabız basıncı
    df["pulse_pressure"] = df["bp_max_first_emerg"] - df["bp_min_first_emerg"]

    return df


def get_lmdb_patient_ids(lmdb_path: Path) -> set[int]:
    """LMDB'deki tüm unique patient ID'lerini döndür."""
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, subdir=True)
    with env.begin() as txn:
        all_keys: list[str] = json.loads(txn.get(b"__keys__").decode())
    env.close()

    patient_ids: set[int] = set()
    for key in all_keys:
        # key format: pXXXXXXXX/sXXXXXXXX/XXXXXXXX_XXXX
        pid_str = key.split("/")[0]  # "p10030053"
        patient_ids.add(int(pid_str.lstrip("p")))

    return patient_ids


def filter_multimodal(df: pd.DataFrame, lmdb_patients: set[int]) -> pd.DataFrame:
    """Sadece LMDB'de X-ray görüntüsü olan hastaları tut."""
    before = len(df)
    df = df[df["patient_id"].isin(lmdb_patients)].reset_index(drop=True)
    after = len(df)
    print(f"  📊 {before:,} → {after:,} hasta  (düşen: {before - after:,})")
    return df


def main() -> None:
    print("=" * 60)
    print("  TabPFN Feature Extraction + Multimodal Filter")
    print("=" * 60 + "\n")

    # 1) Yükle
    print(f"📂 CSV okunuyor: {RAW_CSV.name}")
    df = load_and_select(RAW_CSV)
    print(f"   Satır: {len(df):,}  |  Kolon: {df.shape[1]}\n")

    # 2) Temizle
    print("🧹 Temizleme kuralları uygulanıyor...")
    df = clean(df)
    print(f"   age NaN sayısı : {df['age'].isna().sum()}")
    print(f"   sex NaN sayısı : {df['sex'].isna().sum()}\n")

    # 3) High-NaN kolonları düşür
    print("🗑️  Yüksek NaN kolonlar kontrol ediliyor...")
    df = drop_high_nan_columns(df)
    print()

    # 4) Özellik mühendisliği
    print("⚙️  Hesaplanmış özellikler ekleniyor...")
    df = engineer_features(df)

    # 5) Ara çıktı kaydet (tüm hastalar)
    OUT_RAW.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_RAW, index=False)
    print(f"\n📄 Ara çıktı (tüm hastalar) → {OUT_RAW.name}")
    print(f"   Satır: {len(df):,}  |  Kolon: {df.shape[1]}\n")

    # 6) Multimodal filtre — sadece X-ray'li hastalar
    print("🩻 Multimodal filtre uygulanıyor (LMDB intersection)...")
    lmdb_patients = get_lmdb_patient_ids(LMDB_PATH)
    print(f"  🩻 LMDB'de {len(lmdb_patients):,} unique hasta")
    df = filter_multimodal(df, lmdb_patients)

    # 7) Nihai çıktı kaydet
    OUT_CLEAN.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CLEAN, index=False)
    print(f"\n✅ Kaydedildi → {OUT_CLEAN.name}")
    print(f"   Satır: {len(df):,}  |  Kolon: {df.shape[1]}")

    # 8) Özet istatistikler
    print("\n📊 Özet İstatistikler:")
    print(df.describe().round(2).to_string())

    print("\n" + "=" * 60)
    print("  🎉 Tamamlandı!")
    print("=" * 60)


if __name__ == "__main__":
    main()
