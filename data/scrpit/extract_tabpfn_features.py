"""
TabPFN için özellik çıkarımı — patient_01.csv → tabpfn_features.csv

Kullanım:
    uv run python data/scrpit/extract_tabpfn_features.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "csv" / "patient_01.csv"
OUT_CSV = PROJECT_ROOT / "data" / "processed" / "tabpfn_features.csv"

# ── Kullanılacak ham kolonlar ────────────────────────────────────────────
RAW_COLS = [
    "patient_id",
    "age",
    "sex",
    # First vitals
    "temp_first_emerg",
    "hr_first_emerg",
    "bp_max_first_emerg",
    "bp_min_first_emerg",
    "sat_02_first_emerg",
    "glu_first_emerg",
    # Last vitals
    "temp_last_emerg",
    "sat_02_last_emerg",
    "hr_last_emerg",
    "bp_max_last_emerg",
    "bp_min_last_emerg",
]


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


def main() -> None:
    print("=" * 60)
    print("  TabPFN Feature Extraction")
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

    # 3) Özellik mühendisliği
    print("⚙️  Hesaplanmış özellikler ekleniyor...")
    df = engineer_features(df)

    # 4) Kaydet
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n✅ Kaydedildi → {OUT_CSV}")
    print(f"   Satır: {len(df):,}  |  Kolon: {df.shape[1]}")

    # 5) Özet istatistikler
    print("\n📊 Özet İstatistikler:")
    print(df.describe().round(2).to_string())

    print("\n" + "=" * 60)
    print("  🎉 Tamamlandı!")
    print("=" * 60)


if __name__ == "__main__":
    main()
