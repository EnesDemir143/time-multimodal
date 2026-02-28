from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np

# ── Proje modülünü import et ─────────────────────────────────────────────
from src.data.embedding_cache import (
    EmbeddingCacheDataset,
    EmbeddingCacheWriter,
    RADIOLOGICAL_DIM,
    TABULAR_DIM,
    cache_stats,
    is_cached,
)

# ── Test ayarları ────────────────────────────────────────────────────────
NUM_PATIENTS = 5
PATIENT_IDS = [10030053, 10037793, 10039757, 10082068, 10089641]
LABELS = [1, 1, 0, 0, 0]
# Hasta 0 → 2 X-ray, diğerleri → 1 X-ray
NUM_XRAYS = [2, 1, 1, 1, 1]


def make_dummy_split_csv(splits_dir: Path) -> None:
    """Geçici split CSV oluşturur (tüm hastalar train)."""
    splits_dir.mkdir(parents=True, exist_ok=True)
    csv_path = splits_dir / "fold_0_train.csv"
    with open(csv_path, "w") as f:
        f.write("patient_id,label\n")
        for pid, lbl in zip(PATIENT_IDS, LABELS):
            f.write(f"{pid},{lbl}\n")


def test_write(h5_path: Path) -> dict[str, np.ndarray]:
    """Dummy embedding'leri yazar. Orijinal array'leri döner."""
    rng = np.random.default_rng(42)
    originals: dict[str, np.ndarray] = {}

    with EmbeddingCacheWriter(h5_path) as writer:
        for i, pid in enumerate(PATIENT_IDS):
            # Tabular (fold=0 kullanıyoruz)
            tab = rng.standard_normal(TABULAR_DIM).astype(np.float32)
            writer.save_tabular(pid, tab, fold=0)
            originals[f"tab_p{pid}"] = tab

            # Radiological (bazı hastaların birden fazla X-ray'i var)
            for j in range(NUM_XRAYS[i]):
                rad = rng.standard_normal(RADIOLOGICAL_DIM).astype(np.float32)
                # L2 normalize (RadJEPA gibi)
                rad = rad / np.linalg.norm(rad)
                writer.save_radiological(pid, rad, xray_idx=j)
                originals[f"rad_p{pid}_{j}"] = rad

        # Metadata
        writer.save_metadata(
            patient_ids=np.array(PATIENT_IDS, dtype=np.int64),
            labels=np.array(LABELS, dtype=np.int32),
            num_xrays=np.array(NUM_XRAYS, dtype=np.int32),
        )

    return originals


def test_read_and_verify(
    h5_path: Path,
    splits_dir: Path,
    originals: dict[str, np.ndarray],
) -> None:
    """Cache'ten okur ve orijinallerle karşılaştırır."""
    ds = EmbeddingCacheDataset(
        h5_path=h5_path,
        fold=0,
        split="train",
        splits_dir=splits_dir,
    )

    total_xrays = sum(NUM_XRAYS)
    assert len(ds) == total_xrays, (
        f"Beklenen sample sayısı {total_xrays}, alınan {len(ds)}"
    )

    print(f"   Dataset repr: {ds!r}")
    print(f"   class_counts: {ds.class_counts}")
    print(f"   pos_weight: {ds.pos_weight:.2f}")

    # Her sample'ı doğrula
    for idx in range(len(ds)):
        rad, tab, label = ds[idx]

        # Boyut kontrolü
        assert rad.shape == (RADIOLOGICAL_DIM,), f"Rad boyut hatası idx={idx}: {rad.shape}"
        assert tab.shape == (TABULAR_DIM,), f"Tab boyut hatası idx={idx}: {tab.shape}"

        # Tip kontrolü
        assert rad.dtype == torch.float32, f"Rad dtype hatası: {rad.dtype}"
        assert tab.dtype == torch.float32, f"Tab dtype hatası: {tab.dtype}"

    print("   ✅ Tüm sample'lar doğru boyut ve tipte")

    # Bitwise eşitlik — ilk sample (patient 10030053, xray 0)
    rad_0, tab_0, lbl_0 = ds[0]
    orig_rad = originals["rad_p10030053_0"]
    orig_tab = originals["tab_p10030053"]

    assert np.array_equal(rad_0.numpy(), orig_rad), "Radiological bitwise eşitlik BAŞARISIZ!"
    assert np.array_equal(tab_0.numpy(), orig_tab), "Tabular bitwise eşitlik BAŞARISIZ!"
    assert lbl_0 == 1, f"Label hatası: beklenen 1, alınan {lbl_0}"

    print("   ✅ Bitwise eşitlik doğrulandı (okuma == yazma)")


def test_helpers(h5_path: Path) -> None:
    """is_cached ve cache_stats helper fonksiyonlarını test eder."""
    # is_cached
    assert is_cached(h5_path, 10030053, "radiological"), "is_cached radiological BAŞARISIZ"
    assert is_cached(h5_path, 10030053, "tabular", fold=0), "is_cached tabular BAŞARISIZ"
    assert not is_cached(h5_path, 99999999, "radiological"), "is_cached false-positive!"

    print("   ✅ is_cached helper doğru çalışıyor")

    # cache_stats
    stats = cache_stats(h5_path)
    assert stats["exists"] is True
    assert stats["num_radiological_embeddings"] == sum(NUM_XRAYS)
    assert stats["num_tabular_embeddings_total"] == NUM_PATIENTS
    assert stats["tabular_per_fold"]["fold_0"] == NUM_PATIENTS
    assert stats["num_unique_patients_radiological"] == NUM_PATIENTS

    print(f"   ✅ cache_stats: {stats['num_radiological_embeddings']} rad, "
          f"{stats['num_tabular_embeddings_total']} tab embeddings (fold_0={stats['tabular_per_fold']['fold_0']})")


def test_overwrite_protection(h5_path: Path) -> None:
    """force=False ile aynı key tekrar yazıldığında sessizce atlanır."""
    rng = np.random.default_rng(123)
    new_rad = rng.standard_normal(RADIOLOGICAL_DIM).astype(np.float32)
    new_tab = rng.standard_normal(TABULAR_DIM).astype(np.float32)

    with EmbeddingCacheWriter(h5_path) as writer:
        # force=False (default) — üzerine yazmaz
        writer.save_radiological(10030053, new_rad, xray_idx=0, force=False)
        writer.save_tabular(10030053, new_tab, fold=0, force=False)

    # Orijinal değer hâlâ korunmalı (yeni değer yazılmamış)
    import h5py
    with h5py.File(h5_path, "r") as f:
        stored_rad = f["radiological"]["p10030053_0"][:]
        assert not np.array_equal(stored_rad, new_rad), "force=False ama üzerine yazılmış!"

    print("   ✅ Overwrite koruması çalışıyor (force=False)")


def main() -> None:
    print("=" * 60)
    print("  Embedding Cache (HDF5) Smoke Test")
    print("=" * 60 + "\n")

    tmpdir = Path(tempfile.mkdtemp(prefix="test_embedding_cache_"))
    h5_path = tmpdir / "test_embeddings.h5"
    splits_dir = tmpdir / "splits"

    try:
        # 1) Dummy split oluştur
        print("1️⃣  Dummy split CSV oluşturuluyor...")
        make_dummy_split_csv(splits_dir)
        print(f"   → {splits_dir}/fold_0_train.csv\n")

        # 2) Yazma testi
        print("2️⃣  Embedding'ler yazılıyor...")
        originals = test_write(h5_path)
        print(f"   → {h5_path} ({h5_path.stat().st_size / 1024:.1f} KB)\n")

        # 3) Okuma ve doğrulama
        print("3️⃣  Okuma ve doğrulama...")
        test_read_and_verify(h5_path, splits_dir, originals)
        print()

        # 4) Helper fonksiyonlar
        print("4️⃣  Helper fonksiyonlar...")
        test_helpers(h5_path)
        print()

        # 5) Overwrite koruması
        print("5️⃣  Overwrite koruması...")
        test_overwrite_protection(h5_path)
        print()

        print("=" * 60)
        print("  🎉 Tüm testler başarılı!")
        print("=" * 60)

    finally:
        # Temizlik
        shutil.rmtree(tmpdir, ignore_errors=True)
        print(f"\n🧹 Geçici dizin temizlendi: {tmpdir}")


if __name__ == "__main__":
    import torch  # noqa: E402 — test_read_and_verify'de lazım

    main()
