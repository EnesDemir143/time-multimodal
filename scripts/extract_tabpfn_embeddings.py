"""
TabPFN v2.5 — Hybrid Per-Fold Tabular Embedding Çıkarımı (192-dim).

Her fold için ayrı bir TabPFN modeli eğitilir (yalnızca o fold'un train seti ile).
Sonra bu model, o fold'un **val** setindeki hastalara embedding üretir.
Bu sayede hiçbir validasyon hastasının label bilgisi, kendi embedding'ini
üreten modele sızmaz (data leakage önlemi).

Pipeline (her fold k için):
    fold_k_train.csv → TabPFN fit(X_train, y_train)
    fold_k_val.csv   → TabPFN.get_embeddings(X_val) → 192-dim
    → embeddings.h5  → tabular/fold_{k}/p{pid}

Kullanım:
    PYTHONPATH=. uv run python scripts/extract_tabpfn_embeddings.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import get_config
from src.data.embedding_cache import EmbeddingCacheWriter, TABULAR_DIM, cache_stats
from src.utils import set_seeds

# ── Config ───────────────────────────────────────────────────────────────
_cfg = get_config()

TABULAR_CSV = _cfg.paths.tabpfn_features_clean
H5_PATH = _cfg.paths.embeddings_h5
SPLITS_DIR = _cfg.paths.splits_dir
SEED = _cfg.seed
N_FOLDS = _cfg.cv.n_folds


def load_fold_data(
    fold: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Belirtilen fold'un train ve val verilerini yükle.

    Returns:
        (train_pids, X_train, y_train, val_pids, X_val, y_val)
    """
    # Tüm tabular feature'lar
    tab_df = pd.read_csv(TABULAR_CSV)
    feature_cols = [c for c in tab_df.columns if c != "patient_id"]

    # Fold split CSV'lerini oku
    train_csv = SPLITS_DIR / f"fold_{fold}_train.csv"
    val_csv = SPLITS_DIR / f"fold_{fold}_val.csv"

    train_split = pd.read_csv(train_csv)
    val_split = pd.read_csv(val_csv)

    # Train verisi
    train_pids = train_split["patient_id"].values.astype(int)
    train_labels = train_split["label"].values.astype(np.int64)
    train_tab = tab_df[tab_df["patient_id"].isin(train_pids)]
    # Sıralamayı split CSV ile eşle
    train_tab = train_tab.set_index("patient_id").loc[train_pids].reset_index()
    X_train = train_tab[feature_cols].values.astype(np.float32)

    # Val verisi
    val_pids = val_split["patient_id"].values.astype(int)
    val_labels = val_split["label"].values.astype(np.int64)
    val_tab = tab_df[tab_df["patient_id"].isin(val_pids)]
    val_tab = val_tab.set_index("patient_id").loc[val_pids].reset_index()
    X_val = val_tab[feature_cols].values.astype(np.float32)

    return train_pids, X_train, train_labels, val_pids, X_val, val_labels


def extract_fold_embeddings(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
) -> np.ndarray:
    """Fold'un traini ile TabPFN eğit, val'e embedding üret.

    Args:
        X_train: (N_train, F) train features.
        y_train: (N_train,) train labels.
        X_val: (N_val, F) val features.

    Returns:
        (N_val, 192) float32 numpy array — val embedding'leri.
    """
    from tabpfn import TabPFNClassifier
    from tabpfn_extensions.embedding import TabPFNEmbedding

    clf = TabPFNClassifier(
        n_estimators=1,
        device="cpu",
        ignore_pretraining_limits=True,
    )

    # İç cross-validation ile fit (train verisi içinde)
    embedder = TabPFNEmbedding(tabpfn_clf=clf, n_fold=5)
    embedder.fit(X_train, y_train)

    # Val embedding'leri — data_source="test" çünkü val verisi
    E_val = embedder.get_embeddings(X_train, y_train, X_val, data_source="test")
    # E_val shape: (1, N_val, 192) → squeeze
    E_val = np.squeeze(E_val, axis=0)  # (N_val, 192)

    return E_val.astype(np.float32)


def main() -> None:
    print("=" * 60)
    print("  Sprint 2.2 — TabPFN Hybrid Per-Fold Embedding (192-dim)")
    print("=" * 60 + "\n")

    # 1) Seed
    set_seeds()
    print(f"🌱 Seed: {SEED}")
    print(f"📊 Fold sayısı: {N_FOLDS}\n")

    with EmbeddingCacheWriter(H5_PATH) as writer:
        for fold_k in range(N_FOLDS):
            print(f"{'─' * 50}")
            print(f"🔄 Fold {fold_k}/{N_FOLDS - 1}")
            print(f"{'─' * 50}")

            # 2) Fold verisi yükle
            print(f"  📂 Fold {fold_k} verisi yükleniyor...")
            train_pids, X_train, y_train, val_pids, X_val, y_val = load_fold_data(
                fold_k,
            )
            print(f"     Train: {len(train_pids):,} hasta  |  Val: {len(val_pids):,} hasta")
            print(f"     Train label: Survived={np.sum(y_train == 0)}, Death={np.sum(y_train == 1)}")
            print(f"     Val   label: Survived={np.sum(y_val == 0)}, Death={np.sum(y_val == 1)}")
            print(f"     NaN (train): {np.isnan(X_train).sum():,}  |  NaN (val): {np.isnan(X_val).sum():,}")

            # 3) TabPFN fit (sadece train) → val embedding
            print(f"\n  🧠 TabPFN eğitiliyor (train={len(train_pids):,})...")
            embeddings = extract_fold_embeddings(X_train, y_train, X_val)
            print(f"  ✅ Val embedding shape: {embeddings.shape}")

            # 4) Boyut doğrulama
            assert embeddings.shape == (len(val_pids), TABULAR_DIM), (
                f"Boyut hatası: beklenen ({len(val_pids)}, {TABULAR_DIM}), "
                f"alınan {embeddings.shape}"
            )

            # NaN/Inf kontrolü
            n_nan = np.isnan(embeddings).sum()
            n_inf = np.isinf(embeddings).sum()
            if n_nan > 0 or n_inf > 0:
                print(f"  ⚠️  NaN: {n_nan}  |  Inf: {n_inf}")

            # 5) HDF5'e yaz (fold-aware)
            for i, pid in enumerate(val_pids):
                writer.save_tabular(int(pid), embeddings[i], fold=fold_k)

            print(f"  💾 {len(val_pids):,} val embedding kaydedildi → tabular/fold_{fold_k}/\n")

    # 6) Genel istatistikler
    stats = cache_stats(H5_PATH)
    print("=" * 50)
    print("📊 Cache İstatistikleri:")
    print(f"   Toplam tabular embeddings : {stats['num_tabular_embeddings_total']}")
    print(f"   Tabular per fold:")
    for fk, count in stats.get("tabular_per_fold", {}).items():
        print(f"      {fk}: {count}")
    print(f"   Radiological embeddings   : {stats['num_radiological_embeddings']}")

    print("\n" + "=" * 60)
    print("  🎉 TabPFN Per-Fold embedding çıkarımı tamamlandı!")
    print("=" * 60)


if __name__ == "__main__":
    main()
