# 🔬 LMDB Bitwise Determinism Raporu

> **Tarih:** 2026-02-28 20:59
> **LMDB:** `data/processed/xray.lmdb`
> **Images:** `data/raw/images`

---

## 1. Key Sıralama Testi

> **Amaç:** LMDB'deki `__keys__` metadata listesinin alfabetik sıralı
> olduğunu doğrular. Sıralı key'ler, LMDB B+ tree yapısında verimli
> sequential read sağlar ve DataLoader'ın her epoch'ta tutarlı sırada
> veri okumasını garanti eder.

- Key sayısı: **4,608**
- Sıralı mı: ✅

> **Yorum:** Key'ler sıralıysa LMDB cursor ile sequential okuma
> yapıldığında disk I/O optimum seviyede çalışır. Bu, özellikle
> multi-worker DataLoader kullanırken önemlidir.

## 2. Round-Trip Bütünlük Testi (SHA-256)

> **Amaç:** `data/raw/images/` klasöründeki orijinal JPEG dosyalarının
> byte içeriği ile LMDB'ye yazılmış kopyaları arasında SHA-256 hash
> karşılaştırması yapar. Her görüntü için disk'teki raw bytes okunur,
> aynı key ile LMDB'den okunan bytes ile hash'leri karşılaştırılır.
> Eğer tüm hash'ler eşleşirse, `convert_to_mdb.py` sırasında
> hiçbir byte kaybı/bozulması olmadığı kanıtlanmış olur.

| Metrik | Değer |
|--------|-------|
| Toplam key | 4,608 |
| Eşleşen (SHA-256 identical) | 4,608 |
| Byte uyuşmazlık | 0 |
| Diskte eksik | 0 |
| LMDB'de eksik | 0 |
| **Sonuç** | **✅** |

> **Yorum:** Tüm görüntülerin SHA-256 hash'leri birebir eşleşiyor.
> Bu, LMDB'nin orijinal JPEG byte'larını bozulmadan sakladığını
> ve model eğitiminde kullanılan verilerin kaynak dosyalarla
> tamamen özdeş olduğunu kanıtlar.

## 3. Global Checksum Determinism Testi

> **Amaç:** LMDB'nin her okunuşunda aynı sırayla aynı veriyi
> döndürdüğünü doğrular. Veritabanı 2 kez baştan sona cursor ile
> taranır; her (key, value) çiftinin byte'ları sırayla tek bir
> SHA-256 digest'e beslenir. İki taramanın digest'i aynıysa,
> LMDB okuma sırası deterministiktir — yani aynı seed ile
> aynı epoch sırası garanti edilir.

| Tarama | SHA-256 Digest |
|--------|----------------|
| Run 1 | `e9a40ab1e8be7327...aa372cb4` |
| Run 2 | `e9a40ab1e8be7327...aa372cb4` |

- İki tarama identical: ✅
- Full digest: `e9a40ab1e8be7327892f3132a945e384304c7b87a64cb4c199c946b9aa372cb4`

> **Yorum:** Aynı digest, LMDB'nin B+ tree yapısının her okumada
> aynı key sırasını koruduğunu gösterir. Bu, model eğitiminde
> reproducibility (tekrar üretilebilirlik) için kritiktir —
> aynı veri pipeline'ı farklı makinelerde çalıştırıldığında
> aynı sonuçları üretecektir.

## 4. Metadata Tutarlılık Testi

> **Amaç:** LMDB'deki metadata anahtarlarının (`__len__` ve `__keys__`)
> veritabanındaki gerçek kayıt sayısıyla tutarlı olduğunu doğrular.
> Bu, veri yükleme kodunun doğru kayıt sayısını bilmesini sağlar
> ve eksik/fazla kayıt olup olmadığını tespit eder.

| Metrik | Değer | Durum |
|--------|-------|-------|
| `__len__` | 4,608 | — |
| `len(__keys__)` | 4,608 | ✅ |
| Gerçek kayıt sayısı | 4,608 | ✅ |

> **Yorum:** Üç değerin eşleşmesi, `convert_to_mdb.py`'nin tüm
> görüntüleri eksiksiz yazdığını ve metadata'nın doğru
> güncellendiğini teyit eder. Uyumsuzluk varsa, yazma sırasında
> bir hata oluşmuş demektir.

## 5. LMDB İstatistikleri

> **Amaç:** Veritabanının genel boyut dağılımını ve B+ tree
> yapısını raporlar. Bu bilgiler, disk kullanımını optimize
> etmek ve olası performans sorunlarını tespit etmek için
> kullanılır.

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

> ⏱️ Toplam süre: 109.3 saniye
