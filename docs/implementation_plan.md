# 📋 Implementation Plan — Multimodal Mortalite Tahmini Pipeline'ı

> **Durum:** 🟢 Sprint 0 + 0.5 + Sprint 1 (1.1–1.6) tamamlandı — Sprint 1.7 (Embedding Cache) devam ediyor  
> **Son güncelleme:** 2026-02-28

---

## TL;DR

CDSL göğüs röntgeni + tabular verisinden **binary mortalite tahmini (Death vs Survived)**. Label: `destin_discharge` sütunu (Death=1, Survived=0). Frozen RadJEPA ve TabPFN v2 embedding'leri çıkarılır, projection head ile fuse edilir, tek aşamalı binary classifier ile eğitilir. Tüm süreç deterministik ve reprodüse edilebilir.

---

## Sprint Planı

### Sprint 0 — Pre-Flight *(~1 gün)*

| # | Görev | Çıktı | Durum |
|---|-------|-------|-------|
| 0.1 | EDA: Sınıf dağılımı, sütun yapıları, eksik veri, cross-table analiz | `notebooks/01_data_exploration.ipynb` | ✅ |
| 0.2 | ~~Hierarchical vs Flat karar~~ → **Binary Mortalite (Death vs Survived)** seçildi | `docs/pipeline.md` güncelle | ✅ |
| 0.3 | Eksik veri oranları raporu — NaN analizi notebook'u ile tamamlandı | `notebooks/02_tabpfn_nan_check.ipynb` | ✅ |
| 0.4 | Donanım kontrolü | Aşağıda | ✅ |

> **Donanım:** MacBook Pro — Apple M2 Pro · 16 GB RAM · 80 GB boş disk · Python 3.13.9 · **PyTorch 2.10.0** · **MPS ✅** (`mps:0` aktif) · CUDA ❌ · Veri boyutu: 13 GB

---

### Sprint 0.5 — Veri Ön İşleme *(tamamlandı)* ✅

| # | Görev | Çıktı | Durum |
|---|-------|-------|-------|
| 0.5.1 | Raw CSV + X-ray kopyalama scripti | `scripts/get_xray.py` | ✅ |
| 0.5.2 | TabPFN feature extraction — temizleme & özellik mühendisliği | `scripts/extract_tabpfn_features.py` → `data/processed/tabpfn_features.csv` | ✅ |
| 0.5.3 | X-ray JPEG → LMDB dönüştürme | `scripts/convert_to_mdb.py` → `data/processed/xray.lmdb/` | ✅ |
| 0.5.4 | NaN analizi & yüksek NaN sütun temizliği (%50+ eşik) | `notebooks/02_tabpfn_nan_check.ipynb` → `data/processed/tabpfn_features_clean.csv` | ✅ |

> **Sonuçlar:** `glu_first_emerg` (%99 NaN) silindi. Temiz CSV: 4,479 satır × 17 sütun.

---

### Sprint 1 — Veri Altyapısı *(~2-3 gün)* ✅ (1.1–1.6 tamamlandı)

| # | Görev | Çıktı | Durum |
|---|-------|-------|-------|
| 1.1 | ~~`config/seed.yaml` oluştur~~ | `src/utils.py` içinde `set_seeds()` | ✅ |
| 1.2 | ~~`src/utils/seed.py`~~ — deterministik seed fonksiyonu (Python, NumPy, PyTorch CPU/MPS/CUDA, deterministic flags, single-thread) | `src/utils.py` | ✅ |
| 1.3 | Patient-level StratifiedGroupKFold 5-fold split (mortalite label) | `scripts/create_splits_5fold.py` → `data/splits/5fold/fold_{0-4}_{train,val}.csv` | ✅ |
| 1.4 | Stratification doğrulama scripti (χ² + z-test, md rapor) | `scripts/validate_splits.py` → `docs/split_validation_report.md` | ✅ |
| 1.5 | ~~LMDB cache builder~~ | ~~`src/data/lmdb_builder.py`~~ | ✅ Sprint 0.5'te tamamlandı |
| 1.6 | LMDB bitwise determinizm testi | `scripts/test_lmdb_determinism.py` → `docs/lmdb_determinism_report.md` | ✅ |
| 1.7 | **Embedding Cache Sistemi** — RadJEPA (768-dim) ve TabPFN (192-dim) embedding'lerini bir kez çıkar, `.npy` olarak cache'le. Eğitimde her epoch tekrar çıkarım yapılmaz | `src/data/embedding_cache.py` → `data/embeddings/` | ⬜ |

> [!IMPORTANT]
> **1.7 neden kritik:** RadJEPA her epoch'ta 768-dim çıkarım yaparsa (özellikle CPU'da) eğitim ~3 saat sürer. Cache'lemeden hızlıca dene-yanıl yapılamaz. **Train/val split (1.3) sonrası, embedding çıkarımı (Sprint 2) öncesi yapılmalı.**

---

### Sprint 2 — Embedding Çıkarımı *(~2-3 gün)*

| # | Görev | Çıktı | Durum |
|---|-------|-------|-------|
| 2.1 | `feature_columns.txt` — tabular feature sırası (`tabpfn_features_clean.csv`'den) | Config dosyası | ⬜ |
| 2.2 | TabPFN v2 embedding çıkarımı (192-dim) — girdi: `tabpfn_features_clean.csv` | `data/embeddings/tabular/*.npy` | ⬜ |
| 2.3 | RadJEPA embedding çıkarımı (768-dim) + L2 norm | `data/embeddings/radiological/*.npy` | ⬜ |
| 2.4 | MPS vs CPU determinizm karşılaştırması (3-5 örnek) | Terminal çıktısı | ⬜ |
| 2.5 | Embedding boyut ve nan/inf kontrolü | `scripts/validate_embeddings.py` | ⬜ |
| 2.6 | Hasta bazlı metadata.json oluşturma | `data/embeddings/*/metadata.json` | ⬜ |

---

### Sprint 3 — Model Mimarisi *(~2 gün)*

| # | Görev | Çıktı | Durum |
|---|-------|-------|-------|
| 3.1 | Projection Head (Tabular: 192→64, Vision: 768→128) | `src/models/projection.py` | ⬜ |
| 3.2 | Modality Dropout implementasyonu | `src/models/modality_dropout.py` | ⬜ |
| 3.3 | Parametre sayısı doğrulama (~30-40k) | `scripts/count_params.py` | ⬜ |
| 3.4 | **Binary Classifier** (Death vs Survived, tek aşama, sigmoid çıkış) | `src/models/classifier.py` | ⬜ |

---

### Sprint 4 — Eğitim *(~3-4 gün)*

| # | Görev | Çıktı | Durum |
|---|-------|-------|-------|
| 4.1 | Training loop — Binary Mortalite (Death vs Survived, BCEWithLogitsLoss, pos_weight≈7.14) | `src/training/train.py` | ⬜ |
| 4.2 | Class weight hesaplama utility | `src/utils/class_weights.py` | ⬜ |
| 4.3 | Early stopping mekanizması | `src/training/early_stopping.py` | ⬜ |
| 4.4 | 5-fold cross-validation orchestrator | `src/training/cross_val.py` | ⬜ |
| 4.5 | Val AUROC / Sensitivity @ Specificity > 0.95 kontrolü | Metrik raporu | ⬜ |

---

### Sprint 5 — Robustlaştırma *(~1-2 gün)*

| # | Görev | Çıktı | Durum |
|---|-------|-------|-------|
| 5.1 | Embedding Space Mixup (beta=0.2) | `src/augmentation/mixup.py` | ⬜ |
| 5.2 | Gaussian Noise Regularization | `src/augmentation/noise.py` | ⬜ |
| 5.3 | Agresif Dropout (p=0.6) entegrasyonu | Model dosyasında | ⬜ |

---

### Sprint 6 — Kalibrasyon *(~2 gün)*

| # | Görev | Çıktı | Durum |
|---|-------|-------|-------|
| 6.1 | Temperature Scaling (grid search, fold başına) | `src/calibration/temperature.py` | ⬜ |
| 6.2 | Uncertainty thresholding (reject option) | `src/calibration/reject.py` | ⬜ |
| 6.3 | Metrik hesaplama (Macro-F1, Sens@Spec, ECE) | `src/evaluation/metrics.py` | ⬜ |
| 6.4 | Calibration curve + reliability diagram | `reports/calibration/` | ⬜ |
| 6.5 | Modality ablation çalışması | `reports/ablation/` | ⬜ |

---

### Sprint 7 — Reprodüksiyon ve Paketleme *(~1 gün)*

| # | Görev | Çıktı | Durum |
|---|-------|-------|-------|
| 7.1 | End-to-end determinizm testi (2 çalıştırma karşılaştır) | Test scripti | ⬜ |
| 7.2 | Embedding cache SHA256 manifest | `data/embeddings/manifest.json` | ⬜ |
| 7.3 | `requirements.txt` freeze | Bağımlılık dosyası | ⬜ |
| 7.4 | Nihai dokümantasyon (fold assignments, T değerleri, reject raporu) | `reports/` | ⬜ |

---

## Proje Dizin Yapısı (Güncel)

```
time-multimodal/
├── notebooks/                 # EDA ve analiz notebookları
│   ├── 01_data_exploration.ipynb   # ✅
│   └── 02_tabpfn_nan_check.ipynb   # ✅
├── data/
│   ├── raw/                    # Orijinal CSV + X-ray (dokunma)
│   │   └── images/             # 4,608 JPEG
│   ├── processed/
│   │   ├── tabpfn_features.csv       # ✅
│   │   ├── tabpfn_features_clean.csv # ✅ (4,479 × 17)
│   │   └── xray.lmdb/               # ✅ (~6.5 GB)
│   ├── splits/
│   │   └── 5fold/              # ✅ fold_{0-4}_{train,val}.csv
│   └── embeddings/             # ⬜ Sprint 2'de doldurulacak
│       ├── tabular/
│       └── radiological/
├── docs/
│   ├── pipeline.md                   # ✅
│   ├── implementation_plan.md        # ✅ Bu dosya
│   ├── split_validation_report.md    # ✅
│   └── lmdb_determinism_report.md    # ✅
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py          # re-exports set_seeds
│   │   └── set_seeds.py         # ✅ set_seeds()
│   ├── data/
│   │   ├── embedding_cache.py  # ⬜ Sprint 1.7
│   │   └── dataset.py          # ⬜ Sprint 2+
│   ├── models/
│   │   ├── projection.py       # ⬜ Sprint 3
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
│   ├── create_splits_5fold.py   # ✅ 5-fold split oluşturma
│   ├── validate_splits.py       # ✅ Split doğrulama
│   ├── test_lmdb_determinism.py # ✅ LMDB determinizm testi
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

| Risk | Etki | Olasılık | Mitigasyon |
|------|------|----------|------------|
| Sınıf dengesizliği (Death=550 vs Survived=3,929 → ~7:1) | 🔴 Yüksek | Yüksek | pos_weight≈7.14 + Focal Loss + oversampling |
| MPS non-determinizm | ⚠️ Orta | Yüksek | CPU fallback + deterministik test |
| TabPFN v2 API değişikliği | ⚠️ Orta | Düşük | Versiyon pinle, embed çıktısını cache'le |
| RadJEPA bellek taşması | ⚠️ Orta | Orta | Batch size=1, CPU çıkarım |
| Overfitting (2.5k veri) | 🔴 Yüksek | Yüksek | Dropout(0.6) + Mixup + Noise + Early Stopping |

---

## Sonraki Adımlar

> [!IMPORTANT]
> **Sprint 0, 0.5 ve Sprint 1 (1.1–1.6) tamamlandı.** Split validation ve LMDB determinism raporları `docs/` altında mevcut.
>
> **Sıradaki görev:** Sprint 1.7 — Embedding Cache Sistemi (`src/data/embedding_cache.py`). Ardından Sprint 2 — Frozen Embedding Çıkarımı (TabPFN v2 + RadJEPA).
