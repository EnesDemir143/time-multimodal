# 📊 Split Validation Report

> **Tarih:** 2026-02-28 22:10
> **Anlamlılık düzeyi (α):** 0.05

---

## 1. Dosya Bütünlüğü

> **Amaç:** 5-fold cross-validation için gerekli tüm CSV dosyalarının
> (`fold_X_train.csv`, `fold_X_val.csv`) mevcut olduğunu ve boş
> olmadığını doğrular. Eksik veya boş dosyalar, eğitim pipeline'ının
> hata vermesine neden olur.

| Dosya | Satır | Durum |
|-------|-------|-------|
| `fold_0_train.csv` | 1,454 | ✅ |
| `fold_0_val.csv` | 162 | ✅ |
| `fold_1_train.csv` | 1,454 | ✅ |
| `fold_1_val.csv` | 162 | ✅ |
| `fold_2_train.csv` | 1,454 | ✅ |
| `fold_2_val.csv` | 162 | ✅ |
| `fold_3_train.csv` | 1,454 | ✅ |
| `fold_3_val.csv` | 162 | ✅ |
| `fold_4_train.csv` | 1,454 | ✅ |
| `fold_4_val.csv` | 162 | ✅ |
| `fold_5_train.csv` | 1,455 | ✅ |
| `fold_5_val.csv` | 161 | ✅ |
| `fold_6_train.csv` | 1,454 | ✅ |
| `fold_6_val.csv` | 162 | ✅ |
| `fold_7_train.csv` | 1,455 | ✅ |
| `fold_7_val.csv` | 161 | ✅ |
| `fold_8_train.csv` | 1,455 | ✅ |
| `fold_8_val.csv` | 161 | ✅ |
| `fold_9_train.csv` | 1,455 | ✅ |
| `fold_9_val.csv` | 161 | ✅ |

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
| Fold 5 | 0 | ✅ |
| Fold 6 | 0 | ✅ |
| Fold 7 | 0 | ✅ |
| Fold 8 | 0 | ✅ |
| Fold 9 | 0 | ✅ |

> **Yorum:** Tüm fold'larda kesişim 0 ise, hasta bazlı
> izolasyon sağlanmıştır ve model performans metrikleri
> güvenilirdir.

## 3. Kapsam Kontrolü

> **Amaç:** Tüm hastaların tam olarak bir val fold'unda yer aldığını
> doğrular. Eksik hastalar değerlendirilmemiş veri demektir;
> tekrarlanan hastalar ise performans metriklerini bozar.
> 5-fold CV'de her hasta tam 1 kez val setinde olmalıdır.

- Val'de unique hasta: **1,616** / 1,616 ✅
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

**Global:** 221 Death / 1395 Survived (oran: 0.1368)

| Fold | Train | Train Death% | Val | Val Death% | Sapma |
|------|-------|-------------|-----|-----------|-------|
| 0 | 1,454 | 13.69% | 162 | 13.58% | 0.0010 |
| 1 | 1,454 | 13.69% | 162 | 13.58% | 0.0010 |
| 2 | 1,454 | 13.69% | 162 | 13.58% | 0.0010 |
| 3 | 1,454 | 13.69% | 162 | 13.58% | 0.0010 |
| 4 | 1,454 | 13.62% | 162 | 14.20% | 0.0052 |
| 5 | 1,455 | 13.68% | 161 | 13.66% | 0.0001 |
| 6 | 1,454 | 13.69% | 162 | 13.58% | 0.0010 |
| 7 | 1,455 | 13.68% | 161 | 13.66% | 0.0001 |
| 8 | 1,455 | 13.68% | 161 | 13.66% | 0.0001 |
| 9 | 1,455 | 13.68% | 161 | 13.66% | 0.0001 |

Max sapma: **0.0052** (< 2%) ✅

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
| Val 0 | 140 | 22 |
| Val 1 | 140 | 22 |
| Val 2 | 140 | 22 |
| Val 3 | 140 | 22 |
| Val 4 | 139 | 23 |
| Val 5 | 139 | 22 |
| Val 6 | 140 | 22 |
| Val 7 | 139 | 22 |
| Val 8 | 139 | 22 |
| Val 9 | 139 | 22 |

| Metrik | Değer |
|--------|-------|
| χ² | 0.0437 |
| df | 9 |
| **p-value** | **1.000000** |
| Sonuç | ✅ H₀ reddedilemez → dağılımlar homojen |

> **Yorum:** p-value > α ise, fold'lar arasındaki mortalite oranı
> farkları istatistiksel olarak anlamsızdır (rastgele varyasyondur).
> Bu, stratified split algoritmasının başarıyla çalıştığını doğrular.

### 5.2 Proportion Z-Test (Her Fold vs Global)

> **H₀:** Fold'un mortalite oranı = global oran (0.1368).
> Her fold'un val setindeki mortalite oranı, tek örneklem oran
> z-testi ile global oranla karşılaştırılır. p > α ise o fold'un
> oranı global orandan anlamlı şekilde farklı değildir.

| Fold | Val Death% | z-stat | p-value | Sonuç |
|------|-----------|--------|---------|-------|
| 0 | 22/162 (13.58%) | -0.0354 | 0.971781 | ✅ |
| 1 | 22/162 (13.58%) | -0.0354 | 0.971781 | ✅ |
| 2 | 22/162 (13.58%) | -0.0354 | 0.971781 | ✅ |
| 3 | 22/162 (13.58%) | -0.0354 | 0.971781 | ✅ |
| 4 | 23/162 (14.20%) | +0.1933 | 0.846732 | ✅ |
| 5 | 22/161 (13.66%) | -0.0041 | 0.996716 | ✅ |
| 6 | 22/162 (13.58%) | -0.0354 | 0.971781 | ✅ |
| 7 | 22/161 (13.66%) | -0.0041 | 0.996716 | ✅ |
| 8 | 22/161 (13.66%) | -0.0041 | 0.996716 | ✅ |
| 9 | 22/161 (13.66%) | -0.0041 | 0.996716 | ✅ |

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
