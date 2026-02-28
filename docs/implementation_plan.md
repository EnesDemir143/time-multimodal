# 📋 Implementation Plan — Multimodal Mortalite Tahmini Pipeline'ı

> **Durum:** 🟢 Sprint 0 + 0.5 + Sprint 1 + Sprint 2.1 tamamlandı — Sprint 2.2+ (Embedding Çıkarımı) devam ediyor
> **Son güncelleme:** 2026-02-28

---

## TL;DR

CDSL göğüs röntgeni + tabular verisinden **binary mortalite tahmini (Death vs Survived)**. Label: `destin_discharge` sütunu (Death=1, Survived=0). Frozen RadJEPA ve TabPFN v2 embedding'leri çıkarılır, projection head ile fuse edilir, tek aşamalı binary classifier ile eğitilir. Tüm süreç deterministik ve reprodüse edilebilir.

---

## Sprint Planı

### Sprint 0 — Pre-Flight *(~1 gün)*

| #   | Görev                                                                                     | Çıktı                                | Durum |
| --- | ------------------------------------------------------------------------------------------ | --------------------------------------- | ----- |
| 0.1 | EDA: Sınıf dağılımı, sütun yapıları, eksik veri, cross-table analiz               | `notebooks/01_data_exploration.ipynb` | ✅    |
| 0.2 | ~~Hierarchical vs Flat karar~~ → **Binary Mortalite (Death vs Survived)** seçildi | `docs/pipeline.md` güncelle          | ✅    |
| 0.3 | Eksik veri oranları raporu — NaN analizi notebook'u ile tamamlandı                      | `notebooks/02_tabpfn_nan_check.ipynb` | ✅    |
| 0.4 | Donanım kontrolü                                                                         | Aşağıda                              | ✅    |

> **Donanım:** MacBook Pro — Apple M2 Pro · 16 GB RAM · 80 GB boş disk · Python 3.13.9 · **PyTorch 2.10.0** · **MPS ✅** (`mps:0` aktif) · CUDA ❌ · Veri boyutu: 13 GB

---

### Sprint 0.5 — Veri Ön İşleme *(tamamlandı)* ✅

| #     | Görev                                                           | Çıktı                                                                                | Durum |
| ----- | ---------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ----- |
| 0.5.1 | Raw CSV + X-ray kopyalama scripti                                | `scripts/get_xray.py`                                                                 | ✅    |
| 0.5.2 | TabPFN feature extraction — temizleme & özellik mühendisliği | `scripts/extract_tabpfn_features.py` → `data/processed/tabpfn_features.csv`        | ✅    |
| 0.5.3 | X-ray JPEG → LMDB dönüştürme                                | `scripts/convert_to_mdb.py` → `data/processed/xray.lmdb/`                          | ✅    |
| 0.5.4 | NaN analizi & yüksek NaN sütun temizliği (%50+ eşik)         | `notebooks/02_tabpfn_nan_check.ipynb` → `data/processed/tabpfn_features_clean.csv` | ✅    |

> **Sonuçlar:** `glu_first_emerg` (%99 NaN) silindi. Temiz CSV: 4,479 satır × 17 sütun.

---

### Sprint 1 — Veri Altyapısı *(~2-3 gün)* ✅

| #   | Görev                                                                                                                                | Çıktı                                                                               | Durum                        |
| --- | ------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ---------------------------- |
| 1.1 | ~~`config/seed.yaml` oluştur~~                                                                                                    | `src/utils/set_seeds.py` içinde `set_seeds()`                                     | ✅                           |
| 1.2 | Deterministik seed fonksiyonu (Python, NumPy, PyTorch CPU/MPS/CUDA, deterministic flags, single-thread)                              | `src/utils/set_seeds.py`                                                              | ✅                           |
| 1.3 | Patient-level StratifiedGroupKFold 10-fold split (mortalite label)                                                                     | `scripts/create_splits_5fold.py` → `data/splits/5fold/fold_{0-9}_{train,val}.csv` | ✅                           |
| 1.4 | Stratification doğrulama scripti (χ² + z-test, md rapor)                                                                           | `scripts/validate_splits.py` → `docs/split_validation_report.md`                  | ✅                           |
| 1.5 | ~~LMDB cache builder~~                                                                                                               | ~~`src/data/lmdb_builder.py`~~                                                      | ✅ Sprint 0.5'te tamamlandı |
| 1.6 | LMDB bitwise determinizm testi                                                                                                        | `scripts/test_lmdb_determinism.py` → `docs/lmdb_determinism_report.md`            | ✅                           |
| 1.7 | Embedding Cache sistemi — HDF5'e fold-aware cache'le. Eğitimde her epoch tekrar çıkarım yapılmaz                                           | `src/data/embedding_cache.py` → `data/embeddings/embeddings.h5`                   | ✅                           |

---

### Sprint 2 — Config & Embedding Çıkarımı *(~2-3 gün)*

| #   | Görev                                                                                        | Çıktı                                                     | Durum |
| --- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | ----- |
| 2.1 | Merkezi YAML config sistemi — seed, path, column, dim tüm dosyalarda config'den okunur | `config/config.yaml` + `src/config.py`                     | ✅    |
| 2.2 | TabPFN v2 embedding çıkarımı (192-dim) — **Hybrid Per-Fold Cache** (data leakage önlemi: her fold için ayrı TabPFN fit) | `data/embeddings/embeddings.h5` (`tabular/fold_{k}/`)            | ✅    |
| 2.3 | RadJEPA embedding çıkarımı (768-dim) + L2 norm                                            | `data/embeddings/embeddings.h5` (radiological grubu)       | ✅    |
| 2.4 | MPS vs CPU determinizm karşılaştırması (5 örnek)                                      | `docs/mps_determinism_report.md`                           | ✅    |
| 2.5 | Embedding boyut ve nan/inf kontrolü                                                          | `docs/embedding_validation_report.md`                      | ✅    |
| 2.6 | Hasta bazlı metadata kaydetme (1,616 hasta, label, xray sayıları)                           | `data/embeddings/embeddings.h5` (metadata grubu)           | ✅    |

---

### Sprint 3 — Model Mimarisi *(~2 gün)*

| #   | Görev                                                                         | Çıktı                           | Durum |
| --- | ------------------------------------------------------------------------------ | ---------------------------------- | ----- |
| 3.1 | Projection Head (Tabular: 192→64, Vision: 768→128)                           | `src/models/projection.py`       | ⬜    |
| 3.2 | Modality Dropout implementasyonu                                               | `src/models/modality_dropout.py` | ⬜    |
| 3.3 | Parametre sayısı doğrulama (~30-40k)                                        | `scripts/count_params.py`        | ⬜    |
| 3.4 | **Binary Classifier** (Death vs Survived, tek aşama, sigmoid çıkış) | `src/models/classifier.py`       | ⬜    |

---

### Sprint 4 — Eğitim *(~3-4 gün)*

| #   | Görev                                                                                     | Çıktı                           | Durum |
| --- | ------------------------------------------------------------------------------------------ | ---------------------------------- | ----- |
| 4.1 | Training loop — Binary Mortalite (Death vs Survived, BCEWithLogitsLoss, pos_weight≈7.14) | `src/training/train.py`          | ⬜    |
| 4.2 | Class weight hesaplama utility                                                             | `src/utils/class_weights.py`     | ⬜    |
| 4.3 | Early stopping mekanizması                                                                | `src/training/early_stopping.py` | ⬜    |
| 4.4 | 5-fold cross-validation orchestrator                                                       | `src/training/cross_val.py`      | ⬜    |
| 4.5 | Val AUROC / Sensitivity @ Specificity > 0.95 kontrolü                                     | Metrik raporu                      | ⬜    |

---

### Sprint 5 — Robustlaştırma *(~1-2 gün)*

| #   | Görev                               | Çıktı                      | Durum |
| --- | ------------------------------------ | ----------------------------- | ----- |
| 5.1 | Embedding Space Mixup (beta=0.2)     | `src/augmentation/mixup.py` | ⬜    |
| 5.2 | Gaussian Noise Regularization        | `src/augmentation/noise.py` | ⬜    |
| 5.3 | Agresif Dropout (p=0.6) entegrasyonu | Model dosyasında             | ⬜    |

---

### Sprint 6 — Kalibrasyon *(~2 gün)*

| #   | Görev                                           | Çıktı                           | Durum |
| --- | ------------------------------------------------ | ---------------------------------- | ----- |
| 6.1 | Temperature Scaling (grid search, fold başına) | `src/calibration/temperature.py` | ⬜    |
| 6.2 | Uncertainty thresholding (reject option)         | `src/calibration/reject.py`      | ⬜    |
| 6.3 | Metrik hesaplama (Macro-F1, Sens@Spec, ECE)      | `src/evaluation/metrics.py`      | ⬜    |
| 6.4 | Calibration curve + reliability diagram          | `reports/calibration/`           | ⬜    |
| 6.5 | Modality ablation çalışması                  | `reports/ablation/`              | ⬜    |

---

### Sprint 7 — Reprodüksiyon ve Paketleme *(~1 gün)*

| #   | Görev                                                               | Çıktı                          | Durum |
| --- | -------------------------------------------------------------------- | --------------------------------- | ----- |
| 7.1 | End-to-end determinizm testi (2 çalıştırma karşılaştır)      | Test scripti                      | ⬜    |
| 7.2 | Embedding cache SHA256 manifest                                      | `data/embeddings/manifest.json` | ⬜    |
| 7.3 | `requirements.txt` freeze                                          | Bağımlılık dosyası           | ⬜    |
| 7.4 | Nihai dokümantasyon (fold assignments, T değerleri, reject raporu) | `reports/`                      | ⬜    |

---

## Proje Dizin Yapısı (Güncel)

```
time-multimodal/
├── config/
│   └── config.yaml              # ✅ Merkezi konfigürasyon (Sprint 2.1)
├── notebooks/                   # EDA ve analiz notebookları
│   ├── 01_data_exploration.ipynb   # ✅
│   └── 02_tabpfn_nan_check.ipynb   # ✅
├── data/
│   ├── raw/                     # Orijinal CSV + X-ray (dokunma)
│   │   └── images/              # 4,608 JPEG
│   ├── processed/
│   │   ├── tabpfn_features.csv       # ✅
│   │   ├── tabpfn_features_clean.csv # ✅ (4,479 × 17)
│   │   └── xray.lmdb/               # ✅ (~6.5 GB)
│   ├── splits/
│   │   └── 5fold/               # ✅ fold_{0-9}_{train,val}.csv (10-fold)
│   └── embeddings/              # ⬜ Sprint 2.2+'da doldurulacak
│       └── embeddings.h5        # HDF5:
│                                #   radiological/p{pid}_{idx} (fold-agnostic)
│                                #   tabular/fold_{k}/p{pid}  (per-fold)
│                                #   metadata/
├── docs/
│   ├── pipeline.md                   # ✅
│   ├── implementation_plan.md        # ✅ Bu dosya
│   ├── split_validation_report.md    # ✅
│   ├── lmdb_determinism_report.md    # ✅
│   ├── embedding_validation_report.md # ✅ Sprint 2.5
│   └── mps_determinism_report.md     # ✅ Sprint 2.4
├── src/
│   ├── __init__.py
│   ├── config.py                # ✅ YAML config reader (Sprint 2.1)
│   ├── utils/
│   │   ├── __init__.py          # re-exports set_seeds
│   │   └── set_seeds.py         # ✅ set_seeds() — seed config'den
│   ├── data/
│   │   ├── embedding_cache.py   # ✅ HDF5 fold-aware writer/reader (Sprint 1.7 + Per-Fold)
│   │   └── dataset.py           # ✅ LMDB + tabular dataset
│   ├── models/
│   │   ├── projection.py        # ⬜ Sprint 3
│   │   ├── modality_dropout.py
│   │   └── classifier.py
│   ├── training/
│   │   ├── train.py
│   │   ├── cross_val.py
│   │   └── early_stopping.py
│   ├── augmentation/
│   │   ├── mixup.py
│   │   └── noise.py
│   ├── calibration/
│   │   ├── temperature.py
│   │   └── reject.py
│   └── evaluation/
│       └── metrics.py
├── scripts/
│   ├── get_xray.py              # ✅ Raw veri kopyalama
│   ├── extract_tabpfn_features.py # ✅ Feature extraction
│   ├── convert_to_mdb.py        # ✅ JPEG → LMDB
│   ├── create_splits_5fold.py   # ✅ 10-fold split oluşturma
│   ├── validate_splits.py       # ✅ Split doğrulama
│   ├── test_lmdb_determinism.py # ✅ LMDB determinizm testi
│   ├── test_embedding_cache.py  # ✅ Embedding cache testi
│   ├── validate_embeddings.py   # ⬜ Sprint 2
│   └── count_params.py          # ⬜ Sprint 3
├── reports/
│   ├── calibration/
│   └── ablation/
├── main.py
├── pyproject.toml
└── README.md
```

---

## Risk Analizi

| Risk                                                        | Etki       | Olasılık | Mitigasyon                                    |
| ----------------------------------------------------------- | ---------- | ---------- | --------------------------------------------- |
| Sınıf dengesizliği (Death=550 vs Survived=3,929 → ~7:1) | 🔴 Yüksek | Yüksek    | pos_weight≈7.14 + Focal Loss + oversampling  |
| MPS non-determinizm                                         | ⚠️ Orta  | Yüksek    | CPU fallback + deterministik test             |
| TabPFN v2 API değişikliği                                | ⚠️ Orta  | Düşük   | Versiyon pinle, embed çıktısını cache'le |
| RadJEPA bellek taşması                                    | ⚠️ Orta  | Orta       | Batch size=1, CPU çıkarım                  |
| Overfitting (2.5k veri)                                     | 🔴 Yüksek | Yüksek    | Dropout(0.6) + Mixup + Noise + Early Stopping |

---

## Sonraki Adımlar

> [!IMPORTANT]
> **Sprint 0, 0.5, Sprint 1 ve Sprint 2 tamamlandı.** Tüm embedding'ler `data/embeddings/embeddings.h5`'e cache'lendi.
>
> **Sıradaki görev:** Sprint 3 — Model Mimarisi (Projection Head + Modality Dropout + Binary Classifier).
