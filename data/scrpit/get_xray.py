from __future__ import annotations

import csv
import shutil
from pathlib import Path
from collections import defaultdict

# ── Paths ────────────────────────────────────────────────────────────────
CDSL_ROOT = Path("/Users/enesdemir/Documents/cdsl")
PROJECT_ROOT = Path("/Users/enesdemir/Documents/time-multimodal")
RAW_DIR = PROJECT_ROOT / "data" / "raw"

IMAGES_SRC = CDSL_ROOT / "IMAGES"
DICOM_META = CDSL_ROOT / "CDSL-1.0.0-dicom-metadata.csv"

# CSV dosyaları (kaynak)
CSV_FILES = [
    "patient_01.csv",
    "diagnosis_er_02.csv",
    "diagnosis_hosp_03.csv",
    "vital_signs_04.csv",
    "medication_05.csv",
    "lab_06.csv",
    "atc5.csv",
    "atc7.csv",
    "icd10_codes_dict.txt",
    "CDSL-1.0.0-dicom-metadata.csv",
]

# X-ray modalities (CT hariç)
XRAY_MODALITIES = {"CR", "DX"}


def copy_csvs() -> None:
    """Tüm CSV / metin dosyalarını raw/ klasörüne kopyalar."""
    csv_dir = RAW_DIR / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    for fname in CSV_FILES:
        src = CDSL_ROOT / fname
        dst = csv_dir / fname
        if not src.exists():
            print(f"  ⚠️  Kaynak bulunamadı, atlanıyor: {src}")
            continue
        if dst.exists():
            print(f"  ⏭️  Zaten mevcut, atlanıyor: {dst.name}")
            continue
        print(f"  📄 Kopyalanıyor: {fname}")
        shutil.copy2(src, dst)

    print(f"✅ CSV dosyaları kopyalandı → {csv_dir}\n")


def get_xray_image_paths() -> dict[str, list[str]]:
    """
    DICOM metadata CSV'sini okur, sadece X-ray (CR/DX) olan
    image_id'leri döndürür.

    Returns:
        dict: study_id → [image_id, ...] eşlemesi
    """
    xray_map: dict[str, list[str]] = defaultdict(list)
    total = 0
    skipped_ct = 0

    print("🔍 DICOM metadata okunuyor...")
    with open(DICOM_META, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            modality = row.get("Modality", "").strip()
            if modality not in XRAY_MODALITIES:
                skipped_ct += 1
                continue
            study_id = row.get("study_id", "").strip()
            image_id = row.get("image_id", "").strip()
            patient_folder = row.get("patient_folder_id", "").strip()
            patient_group = row.get("patient_group_folder_id", "").strip()
            if study_id and image_id:
                # Tam kaynak yolunu oluşturmak için tüm bilgiyi saklıyoruz
                key = f"{patient_group}/{patient_folder}/{study_id}"
                xray_map[key].append(image_id)

    n_xray = sum(len(v) for v in xray_map.values())
    print(f"  📊 Toplam kayıt: {total:,}")
    print(f"  🩻 X-ray görüntü: {n_xray:,}")
    print(f"  🚫 Atlanan CT: {skipped_ct:,}")
    print(f"  📁 X-ray study sayısı: {len(xray_map):,}\n")
    return dict(xray_map)


def copy_xray_images(xray_map: dict[str, list[str]]) -> None:
    """X-ray görüntülerini orijinal klasör yapısını koruyarak kopyalar."""
    img_dst_root = RAW_DIR / "images"
    img_dst_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    missing = 0

    print("🩻 X-ray görüntüleri kopyalanıyor...")
    for study_key, image_ids in xray_map.items():
        # study_key = "p10/p10030053/s75704956"
        for image_id in image_ids:
            jpg_name = f"{image_id}.jpg"
            src = IMAGES_SRC / study_key / jpg_name
            dst = img_dst_root / study_key / jpg_name

            if dst.exists():
                skipped += 1
                continue

            if not src.exists():
                missing += 1
                continue

            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1

            if copied % 500 == 0:
                print(f"  ... {copied:,} görüntü kopyalandı")

    print(f"\n✅ X-ray kopyalama tamamlandı → {img_dst_root}")
    print(f"  📋 Kopyalanan: {copied:,}")
    print(f"  ⏭️  Zaten mevcut: {skipped:,}")
    print(f"  ⚠️  Kaynak bulunamadı: {missing:,}")


def main() -> None:
    print("=" * 60)
    print("  CDSL → data/raw  (CSV + X-ray Only)")
    print("=" * 60 + "\n")

    # 1) CSV dosyalarını kopyala
    copy_csvs()

    # 2) X-ray görüntülerini filtrele ve kopyala
    xray_map = get_xray_image_paths()
    copy_xray_images(xray_map)

    print("\n" + "=" * 60)
    print("  🎉 Tamamlandı!")
    print("=" * 60)


if __name__ == "__main__":
    main()
