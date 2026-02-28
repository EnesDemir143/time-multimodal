# 🔬 LMDB Bitwise Determinism Raporu

> **Tarih:** 2026-02-28 20:48
> **LMDB:** `data/processed/xray.lmdb`
> **Images:** `data/raw/images`

---

## 1. Key Sıralama Testi

- Key sayısı: **4,608**
- Sıralı mı: ✅

## 2. Round-Trip Bütünlük Testi (SHA-256)

| Metrik | Değer |
|--------|-------|
| Toplam key | 4,608 |
| Eşleşen (SHA-256 identical) | 4,608 |
| Byte uyuşmazlık | 0 |
| Diskte eksik | 0 |
| LMDB'de eksik | 0 |
| **Sonuç** | **✅** |

## 3. Global Checksum Determinism Testi

| Tarama | SHA-256 Digest |
|--------|----------------|
| Run 1 | `e9a40ab1e8be7327...aa372cb4` |
| Run 2 | `e9a40ab1e8be7327...aa372cb4` |

- İki tarama identical: ✅
- Full digest: `e9a40ab1e8be7327892f3132a945e384304c7b87a64cb4c199c946b9aa372cb4`

## 4. Metadata Tutarlılık Testi

| Metrik | Değer | Durum |
|--------|-------|-------|
| `__len__` | 4,608 | — |
| `len(__keys__)` | 4,608 | ✅ |
| Gerçek kayıt sayısı | 4,608 | ✅ |

## 5. LMDB İstatistikleri

### Görüntü Boyut Dağılımı

| Metrik | Değer |
|--------|-------|
| Toplam görüntü | 4,608 |
| Toplam veri boyutu | 6507.7 MB |
| Ortalama boyut | 1446.2 KB |
| Min boyut | 44.6 KB |
| Max boyut | 6018.7 KB |

### LMDB Veritabanı Bilgileri

| Metrik | Değer |
|--------|-------|
| Disk boyutu | 6544.2 MB |
| Sayfa boyutu | 16,384 bytes |
| Derinlik (B+ tree) | 2 |
| Branch sayfaları | 1 |
| Leaf sayfaları | 15 |
| Overflow sayfaları | 418,810 |
| Toplam entry | 4,610 |
| Map size | 6.4 GiB |

---

## Sonuç

| Test | Durum |
|------|-------|
| Key sıralama | ✅ |
| Round-trip bütünlük (SHA-256) | ✅ |
| Global checksum determinism | ✅ |
| Metadata tutarlılık | ✅ |

**✅ TÜM TESTLER GEÇTİ — LMDB bitwise deterministik**

> ⏱️ Toplam süre: 104.4 saniye
