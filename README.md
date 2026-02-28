# 🏥 TIME-Multimodal

Multimodal mortalite tahmini: Göğüs röntgeni (RadJEPA) ve tabular klinik veri (TabPFN v2) embedding'lerini birleştirerek acil servis hastalarında **binary mortalite sınıflandırması (Death vs Survived)** yapan deterministik bir pipeline.

---

## Mimari

```
┌────────────────────┐      ┌─────────────────────┐
│   X-Ray (JPEG)     │      │  Tabular (CSV)      │
│   via LMDB Cache   │      │  via TabPFN v2      │
└────────┬───────────┘      └────────┬────────────┘
         │                           │
   ┌─────▼─────┐            ┌───────▼────────┐
   │  RadJEPA  │            │  TabPFN v2.5   │
   │  (frozen) │            │  (per-fold)    │
   │  768-dim  │            │  192-dim       │
   └─────┬─────┘            └───────┬────────┘
         │                          │
         └──────────┬───────────────┘
            ┌───────▼────────┐
            │  Fusion Layer  │
            │  (pluggable)   │
            └───────┬────────┘
            ┌───────▼────────┐
            │  Classifier    │
            │  → P(Death)    │
            └────────────────┘
```

> Fusion stratejisi modüler: Concat, Cross-Attention, Gating vb. teknikler denenecek.

## Pipeline İlerlemesi

| Faz | Açıklama | Durum |
|-----|----------|:-----:|
| 0 | Pre-Flight (EDA, donanım kontrolü) | ✅ |
| 0.5 | Veri Ön İşleme (CSV temizleme, LMDB) | ✅ |
| 1 | Veri Altyapısı (seed, splits, cache) | ✅ |
| 2 | Frozen Embedding Çıkarımı | ✅ |
| 3 | Model Mimarisi | ⬜ |
| 4 | Binary Eğitim | ⬜ |
| 5 | Robustlaştırma | ⬜ |
| 6 | Kalibrasyon & Validasyon | ⬜ |
| 7 | Reprodüksiyon & Paketleme | ⬜ |

## Veri Seti

| Metrik | Değer |
|--------|-------|
| Hasta | 1,616 |
| X-Ray | 4,608 (~2.85/hasta) |
| Tabular Feature | 12 |
| Sınıf Dağılımı | Survived: 1,395 · Death: 221 (~6.3:1) |
| Cross-Validation | 10-Fold Stratified |

## Kurulum

```bash
# Python 3.13+ & uv gerekli
uv sync

# Seed, path, fold ayarları
cat config/config.yaml
```

## Embedding Çıkarımı

```bash
# 1) TabPFN — Per-Fold (data leakage önlemi)
PYTHONPATH=. uv run python scripts/extract_tabpfn_embeddings.py

# 2) RadJEPA — Tüm veri, tek seferlik
PYTHONPATH=. uv run python scripts/extract_radjepa_embeddings.py
```

## Proje Yapısı

```
time-multimodal/
├── config/config.yaml           # Merkezi konfigürasyon
├── data/
│   ├── processed/               # Temiz CSV + LMDB
│   ├── splits/5fold/            # 10-fold train/val CSV
│   └── embeddings/embeddings.h5 # Cached embedding'ler (HDF5)
├── src/
│   ├── config.py                # Typed YAML reader
│   ├── data/embedding_cache.py  # HDF5 writer/reader (fold-aware)
│   └── utils/set_seeds.py       # Deterministik seed
├── scripts/                     # Çalıştırılabilir pipeline adımları
└── docs/                        # Detaylı raporlar
```

## Dokümantasyon

| Dosya | İçerik |
|-------|--------|
| [pipeline.md](docs/pipeline.md) | Tüm pipeline tasarımı (FAZ 0-7) |
| [implementation_plan.md](docs/implementation_plan.md) | Sprint bazlı görev takibi |
| [split_validation_report.md](docs/split_validation_report.md) | 10-Fold strateji doğrulama (χ² + z-test) |
| [lmdb_determinism_report.md](docs/lmdb_determinism_report.md) | LMDB bitwise determinizm testi |
| [embedding_validation_report.md](docs/embedding_validation_report.md) | Embedding boyut, NaN/Inf, L2 norm kontrolü |
| [mps_determinism_report.md](docs/mps_determinism_report.md) | MPS vs CPU karşılaştırması |

## Atıflar

Bu projede kullanılan veri seti:

> Ritoré, Á., Oprescu, A. M., Estirado Bronchalo, A., & Armengol de la Hoz, M. Á. (2024). *COVID Data for Shared Learning (CDSL): A comprehensive, multimodal COVID-19 dataset from HM Hospitales* (version 1.0.0). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/1176-6c44

> Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation [Online]*. 101 (23), pp. e215–e220. RRID:SCR_007345.

## Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.
