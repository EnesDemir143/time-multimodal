# 🔬 MPS vs CPU Determinizm Raporu — RadJEPA

> **Tarih:** 2026-02-28  
> **Model:** `AIDElab-IITBombay/RadJEPA`  
> **Test:** 5 örnek görüntü, CPU vs MPS karşılaştırması  
> **Sonuç:** ⚠️ MPS bitwise deterministik değil, CPU deterministik

---

## MPS vs CPU Karşılaştırması (5 Örnek)

| Sample | Key (kısaltılmış) | Bitwise Eşit | Max Fark | Cosine Benzerlik |
|--------|-------------------|:------------:|----------|:----------------:|
| 0 | p10030053/s35028302 | ❌ | 1.57e-05 | 1.00000000 |
| 1 | p10030053/s69266123 | ❌ | 1.97e-05 | 1.00000000 |
| 2 | p10030053/s74847463 | ❌ | 2.80e-05 | 1.00000000 |
| 3 | p10030053/s75704956 | ❌ | 2.09e-05 | 1.00000000 |
| 4 | p10030053/s75704956 | ❌ | 1.51e-05 | 1.00000000 |

**Özet:**
- **Bitwise eşitlik:** ❌ Hiçbir örnekte sağlanmadı
- **Max fark:** ~1.5e-05 – 2.8e-05 (float32 precision sınırında)
- **Cosine similarity:** 1.00000000 (fonksiyonel olarak aynı)

## CPU Determinizm Kontrolü

| Test | Sonuç |
|------|-------|
| CPU Run 1 vs Run 2 (bitwise) | ✅ Birebir aynı |

> [!IMPORTANT]
> **Karar:** Mevcut pipeline MPS kullanıyor (hız avantajı). Çıktılar fonksiyonel olarak deterministik
> (cosine sim = 1.0) ancak bitwise deterministik değil. Tam bitwise reprodüksiyon gerekirse
> `extract_radjepa_embeddings.py`'de device CPU'ya çevrilmeli.
>
> **Mevcut durum:** MPS ile üretilen embedding'ler cache'lendi. Aynı cache tekrar kullanıldığı
> sürece (tekrar çıkarım yapılmadığı sürece) sonuçlar tamamen deterministik.
