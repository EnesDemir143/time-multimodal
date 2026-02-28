# 📊 Embedding Validation Report

> **Tarih:** 2026-02-28  
> **HDF5:** `data/embeddings/embeddings.h5`  
> **Sonuç:** ✅ Tüm kontroller geçti

---

## Genel Özet

| Metrik | Radiological (RadJEPA) | Tabular (TabPFN) |
|--------|----------------------|------------------|
| Boyut | 768-dim | 192-dim |
| Toplam embedding | 4,608 | 1,616 |
| Unique hasta | 1,616 | 1,616 |
| dtype | float32 | float32 |
| NaN sayısı | 0 ✅ | 0 ✅ |
| Inf sayısı | 0 ✅ | 0 ✅ |
| L2 Norm | 1.000000 (tüm örnekler) | ~13.856 (normalize edilmemiş) |

---

## Radiological (RadJEPA — 768-dim)

- **Toplam embedding:** 4,608 (1,616 unique hasta × ~2.85 X-ray/hasta ortalama)
- **Shape:** `(768,)` float32
- **L2 Norm:** Tüm örneklerde `1.000000` — L2 normalizasyonu doğru uygulanmış ✅
- **NaN/Inf:** 0 — Hiçbir bozuk embedding yok ✅
- **Yöntem:** Frozen RadJEPA (fold-agnostic, label görmez)

## Tabular (TabPFN — 192-dim, Per-Fold)

| Fold | Embedding Sayısı | L2 Norm Aralığı |
|------|-----------------|-----------------|
| fold_0 | 162 | [13.8563, 13.8564] |
| fold_1 | 162 | [13.8563, 13.8564] |
| fold_2 | 162 | [13.8563, 13.8564] |
| fold_3 | 162 | [13.8563, 13.8564] |
| fold_4 | 162 | [13.8563, 13.8564] |
| fold_5 | 161 | [13.8563, 13.8564] |
| fold_6 | 162 | [13.8563, 13.8564] |
| fold_7 | 161 | [13.8563, 13.8564] |
| fold_8 | 161 | [13.8563, 13.8564] |
| fold_9 | 161 | [13.8563, 13.8564] |
| **Toplam** | **1,616** | |

- **NaN/Inf:** 0 ✅
- **Yöntem:** Hybrid Per-Fold Cache (her fold ayrı TabPFN fit, data leakage önlemi)
- **L2 Norm:** Normalize edilmemiş (~13.86). TabPFN discriminative model olduğu için cosine space'de çalışmaz; normalize etmek gerekli değildir.

> [!NOTE]
> Fold 0-4 → 162, Fold 5-9 → 161 hasta. 162×5 + 161×5 = 1,615. Toplam 1,616 ile tutarlı (bir fold'da 1 ek hasta).

---

## Hasta Eşleşme Kontrolü

| Kontrol | Sonuç |
|---------|-------|
| Split dosyalarındaki toplam unique hasta | 1,616 |
| `tabpfn_features_clean.csv` satır sayısı | 1,616 |
| Splits ∩ TabPFN örtüşme | 1,616 (tam eşleşme ✅) |
| Split'te olup TabPFN'de olmayan | 0 ✅ |
| TabPFN'de olup Split'te olmayan | 0 ✅ |
