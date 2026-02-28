# 📋 Implementation Plan — Multimodal COVID-19 Tanı Pipeline'ı

> **Durum:** 🟡 Plan aşaması — Henüz implementasyona başlanmadı  
> **Son güncelleme:** 2026-02-28

---

## TL;DR

CDSL göğüs röntgeni + tabular verisinden hiyerarşik (Normal → Pnömoni → COVID/Diğer) sınıflandırma. Frozen RadJEPA ve TabPFN v2 embedding'leri çıkarılır, projection head ile fuse edilir, 2-stage classifier ile eğitilir. Tüm süreç deterministik ve reprodüse edilebilir.

---

## Sprint Planı

### Sprint 0 — Pre-Flight *(~1 gün)*

| # | Görev | Çıktı | Durum |
|---|-------|-------|-------|
| 0.1 | EDA: Sınıf dağılımı, sütun yapıları, eksik veri, cross-table analiz | `notebooks/01_data_exploration.ipynb` | ✅ |
| 0.2 | Hierarchical vs Flat karar ver | `docs/pipeline.md` güncelle | ⬜ |
| 0.3 | Eksik veri oranları raporu | `reports/missing_data.csv` | ⬜ |
| 0.4 | Donanım kontrolü (MPS, disk, RAM) | Terminal çıktısı not edilir | ⬜ |

---

### Sprint 1 — Veri Altyapısı *(~2-3 gün)*

| # | Görev | Çıktı | Durum |
|---|-------|-------|-------|
| 1.1 | `config/seed.yaml` oluştur | Config dosyası | ⬜ |
| 1.2 | `src/utils/seed.py` — deterministik seed fonksiyonu | Python modülü | ⬜ |
| 1.3 | Patient-level GroupKFold 5-fold split | `data/splits/fold_{0-4}_{train,val}.txt` | ⬜ |
| 1.4 | Stratification doğrulama scripti | `scripts/validate_splits.py` | ⬜ |
| 1.5 | LMDB cache builder | `src/data/lmdb_builder.py` | ⬜ |
| 1.6 | LMDB bitwise determinizm testi | `tests/test_lmdb_determinism.py` | ⬜ |

---

### Sprint 2 — Embedding Çıkarımı *(~2-3 gün)*

| # | Görev | Çıktı | Durum |
|---|-------|-------|-------|
| 2.1 | `feature_columns.txt` — tabular feature sırası | Config dosyası | ⬜ |
| 2.2 | TabPFN v2 embedding çıkarımı (192-dim) | `data/embeddings/tabular/*.npy` | ⬜ |
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
| 3.4 | Hierarchical Classifier (Stage 1 + Stage 2) | `src/models/classifier.py` | ⬜ |

---

### Sprint 4 — Eğitim *(~3-4 gün)*

| # | Görev | Çıktı | Durum |
|---|-------|-------|-------|
| 4.1 | Training loop — Stage 1 (Binary: Normal vs Pnömoni) | `src/training/stage1.py` | ⬜ |
| 4.2 | Training loop — Stage 2 (COVID vs Diğer) | `src/training/stage2.py` | ⬜ |
| 4.3 | Class weight hesaplama utility | `src/utils/class_weights.py` | ⬜ |
| 4.4 | Early stopping mekanizması | `src/training/early_stopping.py` | ⬜ |
| 4.5 | 5-fold cross-validation orchestrator | `src/training/cross_val.py` | ⬜ |
| 4.6 | Stage 1 sonrası val specificity > 0.95 kontrolü | Metrik raporu | ⬜ |

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

## Proje Dizin Yapısı (Önerilen)

```
time-multimodal/
├── config/
│   └── seed.yaml
├── notebooks/                 # EDA ve analiz notebookları
│   └── 01_data_exploration.ipynb
├── data/
│   ├── raw/                    # Orijinal veri (dokunma)
│   ├── splits/                 # Fold dosyaları
│   ├── lmdb/                   # Cached görüntüler
│   └── embeddings/             # Frozen embedding'ler
│       ├── tabular/
│       └── radiological/
├── docs/
│   ├── pipeline.md             # ✅ Bu dosya
│   └── implementation_plan.md  # ✅ Bu dosya
├── src/
│   ├── data/
│   │   ├── lmdb_builder.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── projection.py
│   │   ├── modality_dropout.py
│   │   └── classifier.py
│   ├── training/
│   │   ├── stage1.py
│   │   ├── stage2.py
│   │   ├── cross_val.py
│   │   └── early_stopping.py
│   ├── augmentation/
│   │   ├── mixup.py
│   │   └── noise.py
│   ├── calibration/
│   │   ├── temperature.py
│   │   └── reject.py
│   ├── evaluation/
│   │   └── metrics.py
│   └── utils/
│       ├── seed.py
│       └── class_weights.py
├── scripts/
│   ├── validate_splits.py
│   ├── validate_embeddings.py
│   └── count_params.py
├── tests/
│   └── test_lmdb_determinism.py
├── reports/
│   ├── eda_report.md
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
| Diğer Pnömoni < 100 örnek | ⚠️ Yüksek | Orta | Hierarchical + ağır class weighting |
| MPS non-determinizm | ⚠️ Orta | Yüksek | CPU fallback + deterministik test |
| TabPFN v2 API değişikliği | ⚠️ Orta | Düşük | Versiyon pinle, embed çıktısını cache'le |
| RadJEPA bellek taşması | ⚠️ Orta | Orta | Batch size=1, CPU çıkarım |
| Overfitting (2.5k veri) | 🔴 Yüksek | Yüksek | Dropout(0.6) + Mixup + Noise + Early Stopping |

---

## Sonraki Adım

> [!IMPORTANT]
> **Implementasyona başlamadan önce FAZ 0 (EDA) tamamlanmalı.** Diğer Pnömoni sınıfının örnek sayısı tüm stratejiyi belirler.
