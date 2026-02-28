"""
X-ray JPEG → LMDB dönüştürücü

Kullanım:
    uv add lmdb tqdm
    uv run python data/scrpit/convert_to_mdb.py
"""
from __future__ import annotations

import json
from pathlib import Path

import lmdb
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "images"
LMDB_PATH = PROJECT_ROOT / "data" / "processed" / "xray.lmdb"

# LMDB map size — 50 GB (yeterli headroom)
MAP_SIZE = 50 * 1024 * 1024 * 1024  # 50 GiB


def collect_image_paths(root: Path) -> list[tuple[str, Path]]:
    """
    Tüm JPEG dosyalarını toplar ve (key, path) listesi döndürür.

    Key formatı: patient_id/study_id/image_file
    Örn: p10030053/s74847463/74847463_0001
    """
    entries: list[tuple[str, Path]] = []

    for jpg_path in sorted(root.rglob("*.jpg")):
        # Yapı: images/pXX/pXXXXXXXX/sXXXXXXXX/XXXXXXXX_XXXX.jpg
        parts = jpg_path.relative_to(root).parts
        # parts = ("p10", "p10030053", "s74847463", "74847463_0001.jpg")
        if len(parts) < 4:
            continue

        patient_id = parts[1]  # p10030053
        study_id = parts[2]    # s74847463
        image_stem = jpg_path.stem  # 74847463_0001

        key = f"{patient_id}/{study_id}/{image_stem}"
        entries.append((key, jpg_path))

    return entries


def build_lmdb(entries: list[tuple[str, Path]], lmdb_path: Path) -> None:
    """Tüm JPEG'leri LMDB veritabanına yazar."""
    lmdb_path.parent.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(
        str(lmdb_path),
        map_size=MAP_SIZE,
        subdir=True,
        readonly=False,
        lock=False,
        meminit=False,
        map_async=True,
    )

    keys: list[str] = []

    print("💾 LMDB'ye yazılıyor...")
    with env.begin(write=True) as txn:
        for key, jpg_path in tqdm(entries, desc="  Görüntüler", unit="img"):
            raw_bytes = jpg_path.read_bytes()
            txn.put(key.encode("utf-8"), raw_bytes)
            keys.append(key)

        # Metadata anahtarları
        txn.put(b"__keys__", json.dumps(keys).encode("utf-8"))
        txn.put(b"__len__", str(len(keys)).encode("utf-8"))

    env.sync()
    env.close()


def main() -> None:
    print("=" * 60)
    print("  X-ray JPEG → LMDB Conversion")
    print("=" * 60 + "\n")

    # 1) Görüntüleri topla
    print(f"🔍 Görüntüler taranıyor: {IMAGES_DIR}")
    entries = collect_image_paths(IMAGES_DIR)
    print(f"   Bulunan görüntü: {len(entries):,}\n")

    if not entries:
        print("⚠️  Hiç görüntü bulunamadı!")
        return

    # 2) LMDB oluştur
    build_lmdb(entries, LMDB_PATH)

    # 3) Doğrulama
    print(f"\n✅ LMDB oluşturuldu → {LMDB_PATH}")
    env = lmdb.open(str(LMDB_PATH), readonly=True, lock=False, subdir=True)
    with env.begin() as txn:
        n = int(txn.get(b"__len__").decode())
        stat = env.stat()
    env.close()

    print(f"   Toplam kayıt  : {n:,}")
    print(f"   DB sayfa sayısı: {stat['leaf_pages']:,}")

    # Disk boyutu
    total_bytes = sum(f.stat().st_size for f in LMDB_PATH.iterdir())
    print(f"   Disk boyutu   : {total_bytes / (1024**2):.1f} MB")

    print("\n" + "=" * 60)
    print("  🎉 Tamamlandı!")
    print("=" * 60)


if __name__ == "__main__":
    main()
