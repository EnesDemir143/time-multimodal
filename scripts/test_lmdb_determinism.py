"""
LMDB Bitwise Determinism Testi

Kullanım:
    uv run python scripts/test_lmdb_determinism.py
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import lmdb
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "images"
LMDB_PATH = PROJECT_ROOT / "data" / "processed" / "xray.lmdb"
REPORT_DIR = PROJECT_ROOT / "docs"
REPORT_PATH = REPORT_DIR / "lmdb_determinism_report.md"

PASS = "✅"
FAIL = "❌"


# ── Yardımcı ─────────────────────────────────────────────────────────────
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def collect_image_paths(root: Path) -> list[tuple[str, Path]]:
    """convert_to_mdb.py ile aynı mantık — deterministik key üretimi."""
    entries: list[tuple[str, Path]] = []
    for jpg_path in sorted(root.rglob("*.jpg")):
        parts = jpg_path.relative_to(root).parts
        if len(parts) < 4:
            continue
        patient_id = parts[1]
        study_id = parts[2]
        image_stem = jpg_path.stem
        key = f"{patient_id}/{study_id}/{image_stem}"
        entries.append((key, jpg_path))
    return entries


# ── Test Fonksiyonları ───────────────────────────────────────────────────
def test_key_ordering(env: lmdb.Environment, log):
    """Test 1: __keys__ listesinin sıralı olduğunu doğrula."""
    log("## 1. Key Sıralama Testi")
    log("")

    with env.begin() as txn:
        raw = txn.get(b"__keys__")
        keys: list[str] = json.loads(raw.decode("utf-8"))

    sorted_keys = sorted(keys)
    is_sorted = keys == sorted_keys
    n_keys = len(keys)

    log(f"- Key sayısı: **{n_keys:,}**")
    log(f"- Sıralı mı: {PASS if is_sorted else FAIL}")
    if not is_sorted:
        # İlk sırasız key'i bul
        for i in range(len(keys) - 1):
            if keys[i] > keys[i + 1]:
                log(f"- İlk sırasız index: **{i}** → `{keys[i]}` > `{keys[i+1]}`")
                break
    log("")
    return is_sorted, keys


def test_roundtrip_integrity(env: lmdb.Environment, keys: list[str], log):
    """Test 2: Diskteki JPEG bytes ile LMDB'deki bytes SHA-256 karşılaştırması."""
    log("## 2. Round-Trip Bütünlük Testi (SHA-256)")
    log("")

    mismatches: list[str] = []
    missing_disk: list[str] = []
    missing_lmdb: list[str] = []

    disk_entries = collect_image_paths(IMAGES_DIR)
    disk_map = {k: p for k, p in disk_entries}

    with env.begin() as txn:
        for key in tqdm(keys, desc="  Round-trip kontrol", unit="img"):
            lmdb_val = txn.get(key.encode("utf-8"))
            if lmdb_val is None:
                missing_lmdb.append(key)
                continue

            if key not in disk_map:
                missing_disk.append(key)
                continue

            disk_bytes = disk_map[key].read_bytes()
            if sha256_bytes(disk_bytes) != sha256_bytes(lmdb_val):
                mismatches.append(key)

    total = len(keys)
    n_ok = total - len(mismatches) - len(missing_disk) - len(missing_lmdb)
    all_ok = len(mismatches) == 0 and len(missing_disk) == 0 and len(missing_lmdb) == 0

    log("| Metrik | Değer |")
    log("|--------|-------|")
    log(f"| Toplam key | {total:,} |")
    log(f"| Eşleşen (SHA-256 identical) | {n_ok:,} |")
    log(f"| Byte uyuşmazlık | {len(mismatches)} |")
    log(f"| Diskte eksik | {len(missing_disk)} |")
    log(f"| LMDB'de eksik | {len(missing_lmdb)} |")
    log(f"| **Sonuç** | **{PASS if all_ok else FAIL}** |")
    log("")

    if mismatches:
        log(f"> ⚠️ İlk 5 uyuşmazlık: {mismatches[:5]}")
        log("")

    return all_ok


def test_global_checksum(env: lmdb.Environment, log):
    """Test 3: Aynı sıralı taramayı 2 kez yap, digest'lerin aynı olduğunu doğrula."""
    log("## 3. Global Checksum Determinism Testi")
    log("")

    digests: list[str] = []
    for run in range(2):
        h = hashlib.sha256()
        with env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                if key.startswith(b"__"):
                    continue
                h.update(key)
                h.update(value)
        digests.append(h.hexdigest())

    identical = digests[0] == digests[1]

    log("| Tarama | SHA-256 Digest |")
    log("|--------|----------------|")
    log(f"| Run 1 | `{digests[0][:16]}...{digests[0][-8:]}` |")
    log(f"| Run 2 | `{digests[1][:16]}...{digests[1][-8:]}` |")
    log("")
    log(f"- İki tarama identical: {PASS if identical else FAIL}")
    log(f"- Full digest: `{digests[0]}`")
    log("")

    return identical


def test_metadata_consistency(env: lmdb.Environment, log):
    """Test 4: __len__ ve __keys__ metadata tutarlılığı."""
    log("## 4. Metadata Tutarlılık Testi")
    log("")

    with env.begin() as txn:
        raw_len = txn.get(b"__len__")
        raw_keys = txn.get(b"__keys__")

        stored_len = int(raw_len.decode("utf-8"))
        keys_list = json.loads(raw_keys.decode("utf-8"))
        keys_len = len(keys_list)

        # Gerçek kayıt sayısını say (metadata hariç)
        actual_count = 0
        cursor = txn.cursor()
        for k, _ in cursor:
            if not k.startswith(b"__"):
                actual_count += 1

    len_match = stored_len == keys_len
    count_match = stored_len == actual_count
    all_ok = len_match and count_match

    log("| Metrik | Değer | Durum |")
    log("|--------|-------|-------|")
    log(f"| `__len__` | {stored_len:,} | — |")
    log(f"| `len(__keys__)` | {keys_len:,} | {PASS if len_match else FAIL} |")
    log(f"| Gerçek kayıt sayısı | {actual_count:,} | {PASS if count_match else FAIL} |")
    log("")

    return all_ok


def test_statistics(env: lmdb.Environment, keys: list[str], log):
    """Test 5: LMDB istatistikleri."""
    log("## 5. LMDB İstatistikleri")
    log("")

    sizes: list[int] = []
    with env.begin() as txn:
        for key in tqdm(keys, desc="  İstatistik toplama", unit="img"):
            val = txn.get(key.encode("utf-8"))
            if val is not None:
                sizes.append(len(val))

    stat = env.stat()
    info = env.info()

    total_bytes = sum(sizes)
    avg_size = total_bytes / len(sizes) if sizes else 0
    min_size = min(sizes) if sizes else 0
    max_size = max(sizes) if sizes else 0

    # Disk boyutu
    disk_bytes = sum(f.stat().st_size for f in LMDB_PATH.iterdir())

    log("### Görüntü Boyut Dağılımı")
    log("")
    log("| Metrik | Değer |")
    log("|--------|-------|")
    log(f"| Toplam görüntü | {len(sizes):,} |")
    log(f"| Toplam veri boyutu | {total_bytes / (1024**2):.1f} MB |")
    log(f"| Ortalama boyut | {avg_size / 1024:.1f} KB |")
    log(f"| Min boyut | {min_size / 1024:.1f} KB |")
    log(f"| Max boyut | {max_size / 1024:.1f} KB |")
    log("")

    log("### LMDB Veritabanı Bilgileri")
    log("")
    log("| Metrik | Değer |")
    log("|--------|-------|")
    log(f"| Disk boyutu | {disk_bytes / (1024**2):.1f} MB |")
    log(f"| Sayfa boyutu | {stat['psize']:,} bytes |")
    log(f"| Derinlik (B+ tree) | {stat['depth']} |")
    log(f"| Branch sayfaları | {stat['branch_pages']:,} |")
    log(f"| Leaf sayfaları | {stat['leaf_pages']:,} |")
    log(f"| Overflow sayfaları | {stat['overflow_pages']:,} |")
    log(f"| Toplam entry | {stat['entries']:,} |")
    log(f"| Map size | {info['map_size'] / (1024**3):.1f} GiB |")
    log("")

    return True  # istatistik her zaman geçer


# ── Ana ──────────────────────────────────────────────────────────────────
def main() -> None:
    lines: list[str] = []

    def log(text: str = "") -> None:
        print(text)
        lines.append(text)

    log("# 🔬 LMDB Bitwise Determinism Raporu")
    log("")
    log(f"> **Tarih:** {time.strftime('%Y-%m-%d %H:%M')}")
    log(f"> **LMDB:** `{LMDB_PATH.relative_to(PROJECT_ROOT)}`")
    log(f"> **Images:** `{IMAGES_DIR.relative_to(PROJECT_ROOT)}`")
    log("")
    log("---")
    log("")

    # LMDB aç
    env = lmdb.open(str(LMDB_PATH), readonly=True, lock=False, subdir=True)

    # ── Testler ──────────────────────────────────────────────────
    t0 = time.time()

    ok_sort, keys = test_key_ordering(env, log)
    ok_roundtrip = test_roundtrip_integrity(env, keys, log)
    ok_checksum = test_global_checksum(env, log)
    ok_meta = test_metadata_consistency(env, log)
    test_statistics(env, keys, log)

    elapsed = time.time() - t0
    env.close()

    # ── Genel Sonuç ──────────────────────────────────────────────
    log("---")
    log("")
    log("## Sonuç")
    log("")

    all_ok = ok_sort and ok_roundtrip and ok_checksum and ok_meta

    log("| Test | Durum |")
    log("|------|-------|")
    log(f"| Key sıralama | {PASS if ok_sort else FAIL} |")
    log(f"| Round-trip bütünlük (SHA-256) | {PASS if ok_roundtrip else FAIL} |")
    log(f"| Global checksum determinism | {PASS if ok_checksum else FAIL} |")
    log(f"| Metadata tutarlılık | {PASS if ok_meta else FAIL} |")
    log("")
    log(f"**{'✅ TÜM TESTLER GEÇTİ — LMDB bitwise deterministik' if all_ok else '❌ BAZI TESTLER BAŞARISIZ'}**")
    log("")
    log(f"> ⏱️ Toplam süre: {elapsed:.1f} saniye")

    # Raporu kaydet
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n📄 Rapor kaydedildi → {REPORT_PATH}")


if __name__ == "__main__":
    main()
