# 📊 Split Validation Report

> **Tarih:** 2026-02-28 20:59
> **Anlamlılık düzeyi (α):** 0.05

---

## 1. Dosya Bütünlüğü

> **Amaç:** 5-fold cross-validation için gerekli tüm CSV dosyalarının
> (`fold_X_train.csv`, `fold_X_val.csv`) mevcut olduğunu ve boş
> olmadığını doğrular. Eksik veya boş dosyalar, eğitim pipeline'ının
> hata vermesine neden olur.

| Dosya | Satır | Durum |
|-------|-------|-------|
| `fold_0_train.csv` | 3,583 | ✅ |
| `fold_0_val.csv` | 896 | ✅ |
| `fold_1_train.csv` | 3,583 | ✅ |
| `fold_1_val.csv` | 896 | ✅ |
| `fold_2_train.csv` | 3,583 | ✅ |
| `fold_2_val.csv` | 896 | ✅ |
| `fold_3_train.csv` | 3,584 | ✅ |
| `fold_3_val.csv` | 895 | ✅ |
| `fold_4_train.csv` | 3,583 | ✅ |
| `fold_4_val.csv` | 896 | ✅ |

> **Yorum:** Tüm fold dosyalarının mevcut ve dolu olması, split
> işleminin başarıyla tamamlandığını gösterir.

## 2. Data Leakage Kontrolü

> **Amaç:** Aynı hastanın hem train hem de val setinde yer alıp
> almadığını kontrol eder. Data leakage, modelin val setindeki
> hastaları eğitim sırasında görmesine neden olur ve gerçek
> performansın olduğundan yüksek görünmesine yol açar.
> Bu, klinik AI çalışmalarında en kritik hatalardan biridir.

| Fold | train ∩ val | Durum |
|------|-------------|-------|
| Fold 0 | 0 | ✅ |
| Fold 1 | 0 | ✅ |
| Fold 2 | 0 | ✅ |
| Fold 3 | 0 | ✅ |
| Fold 4 | 0 | ✅ |

> **Yorum:** Tüm fold'larda kesişim 0 ise, hasta bazlı
> izolasyon sağlanmıştır ve model performans metrikleri
> güvenilirdir.

## 3. Kapsam Kontrolü

> **Amaç:** Tüm hastaların tam olarak bir val fold'unda yer aldığını
> doğrular. Eksik hastalar değerlendirilmemiş veri demektir;
> tekrarlanan hastalar ise performans metriklerini bozar.
> 5-fold CV'de her hasta tam 1 kez val setinde olmalıdır.

- Val'de unique hasta: **4,479** / 4,479 ✅
- Tekrarlanan hasta (val'ler arası): **0** ✅

> **Yorum:** 5 fold'un val setleri birleştirildiğinde tüm
> hastaları kapsıyor ve hiçbir hasta birden fazla fold'da
> tekrarlanmıyorsa, cross-validation yapısı doğrudur.

## 4. Stratifikasyon Dağılımı

> **Amaç:** Her fold'daki mortalite (ölüm) oranının global orandan
> ne kadar saptığını ölçer. Stratified split, sınıf dengesini
> korumayı hedefler. <%2 sapma kabul edilebilir aralıktadır.
> Dengesiz fold'lar, modelin bazı fold'larda sistematik olarak
> farklı performans göstermesine neden olabilir.

**Global:** 550 Death / 3929 Survived (oran: 0.1228)

| Fold | Train | Train Death% | Val | Val Death% | Sapma |
|------|-------|-------------|-----|-----------|-------|
| 0 | 3,583 | 12.28% | 896 | 12.28% | 0.0000 |
| 1 | 3,583 | 12.28% | 896 | 12.28% | 0.0000 |
| 2 | 3,583 | 12.28% | 896 | 12.28% | 0.0000 |
| 3 | 3,584 | 12.28% | 895 | 12.29% | 0.0001 |
| 4 | 3,583 | 12.28% | 896 | 12.28% | 0.0000 |

Max sapma: **0.0001** (< 2%) ✅

> **Yorum:** Max sapma <%2 ise fold'lar arasında sınıf dağılımı
> dengeli demektir. Bu, cross-validation sonuçlarının fold
> seçimine bağlı olmadığını gösterir.

## 5. İstatistiksel Testler

> **Amaç:** Fold'lar arası dağılım farklılıklarının istatistiksel
> olarak anlamlı olup olmadığını test eder. Gözle görülen küçük
> farkların rastgele mi yoksa sistematik mi olduğunu ayırt etmek
> için hipotez testleri kullanılır.

### 5.1 Chi-Squared Homojenlik Testi

> **H₀:** Tüm fold'lardaki mortalite oranları aynı dağılımdan geliyor.
> Bu test, 5 fold'un val setlerindeki survived/death sayılarını bir
> contingency tablosuna yerleştirir ve χ² testi uygular. p > α ise
> dağılımlar homojendir (fold'lar arası anlamlı fark yoktur).

| Fold | Survived | Death |
|------|----------|-------|
| Val 0 | 786 | 110 |
| Val 1 | 786 | 110 |
| Val 2 | 786 | 110 |
| Val 3 | 785 | 110 |
| Val 4 | 786 | 110 |

| Metrik | Değer |
|--------|-------|
| χ² | 0.0001 |
| df | 4 |
| **p-value** | **1.000000** |
| Sonuç | ✅ H₀ reddedilemez → dağılımlar homojen |

> **Yorum:** p-value > α ise, fold'lar arasındaki mortalite oranı
> farkları istatistiksel olarak anlamsızdır (rastgele varyasyondur).
> Bu, stratified split algoritmasının başarıyla çalıştığını doğrular.

### 5.2 Proportion Z-Test (Her Fold vs Global)

> **H₀:** Fold'un mortalite oranı = global oran (0.1228).
> Her fold'un val setindeki mortalite oranı, tek örneklem oran
> z-testi ile global oranla karşılaştırılır. p > α ise o fold'un
> oranı global orandan anlamlı şekilde farklı değildir.

| Fold | Val Death% | z-stat | p-value | Sonuç |
|------|-----------|--------|---------|-------|
| 0 | 110/896 (12.28%) | -0.0025 | 0.998005 | ✅ |
| 1 | 110/896 (12.28%) | -0.0025 | 0.998005 | ✅ |
| 2 | 110/896 (12.28%) | -0.0025 | 0.998005 | ✅ |
| 3 | 110/895 (12.29%) | +0.0100 | 0.992017 | ✅ |
| 4 | 110/896 (12.28%) | -0.0025 | 0.998005 | ✅ |

✅ **Tüm fold'lar global oranla istatistiksel olarak uyumlu (p > 0.05)**

> **Yorum:** Tüm fold'larda p > α ise, hiçbir fold'un mortalite
> oranı global populasyondan istatistiksel olarak sapmıyordur.
> Bu, her fold'un genel veri setini temsil ettiğini kanıtlar.

---

## Sonuç

| Kontrol | Durum |
|---------|-------|
| Dosya bütünlüğü | ✅ |
| Data leakage | ✅ |
| Kapsam | ✅ |
| Stratifikasyon (< 2%) | ✅ |
| χ² homojenlik (p > 0.05) | ✅ |
| Z-test tüm fold'lar (p > 0.05) | ✅ |

**✅ TÜM KONTROLLER GEÇTİ**
