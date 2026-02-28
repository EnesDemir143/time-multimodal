# 🏥 Multimodal COVID-19 Tanı Pipeline'ı

> **Proje**: CDSL Göğüs Röntgeni + Tabular Veri ile Hiyerarşik COVID-19 Sınıflandırma  
> **Mimari**: Frozen RadJEPA (Görüntü) + Frozen TabPFN v2 (Tabular) → Projection Head → Hierarchical Classifier  
> **Hedef**: Deterministik, reprodüse edilebilir, klinik güvenilirlikte bir tanı sistemi

---

## İçindekiler

1. [FAZ 0 — Pre-Flight Check](#faz-0--pre-flight-check)
2. [FAZ 1 — Veri Altyapısı ve Determinizm](#faz-1--veri-altyapısı-ve-determinizm)
3. [FAZ 2 — Frozen Embedding Çıkarımı](#faz-2--frozen-embedding-çıkarımı)
4. [FAZ 3 — Boyut İndirgeme ve Fusion Mimarisi](#faz-3--boyut-indirgeme-ve-fusion-mimarisi)
5. [FAZ 4 — Hiyerarşik Eğitim Protokolü](#faz-4--hiyerarşik-eğitim-protokolü)
6. [FAZ 5 — Robustlaştırma ve Augmentasyon](#faz-5--robustlaştırma-ve-augmentasyon)
7. [FAZ 6 — Kalibrasyon ve Klinik Validasyon](#faz-6--kalibrasyon-ve-klinik-validasyon)
8. [FAZ 7 — Reprodüksiyon ve Paketleme](#faz-7--reprodüksiyon-ve-paketleme)
9. [Sorun-Çözüm Tablosu](#sorun-çözüm-tablosu)

---

## FAZ 0 — Pre-Flight Check

### 0.1 Exploratory Data Analysis (EDA)

Veri setine dokunmadan önce sınıf dağılımını analiz et.

**Class Histogramı çıkar:**

| Sınıf | ICD Kodu | Min. Beklenen |
|-------|----------|---------------|
| Normal | — | ≥300 |
| COVID-19 | U07.1 | ≥300 |
| Diğer Pnömoni | J12-J18 | Kontrol et |

**Karar Noktası:**

| Durum | Strateji |
|-------|----------|
| Diğer Pnömoni > 300 örnek | Flat 3-class classifier yeterli |
| Diğer Pnömoni < 300 örnek | **Hierarchical Strategy zorunlu** (pipeline varsayımı) |

**Eksik Veri Analizi:**
- Hangi hastalarda tabular veri eksik?
- Hangi hastalarda görüntü eksik?
- Oranları not et → %20+ eksik varsa **modality dropout** daha kritik hale gelir.

### 0.2 Donanım ve Ortam Kontrolü

| Kontrol | Komut / Yöntem | Beklenen |
|---------|----------------|----------|
| MPS kullanılabilirliği | `torch.backends.mps.is_available()` | `True` (M2) |
| Disk alanı | LMDB cache için | ~50GB boş alan |
| RAM | RadJEPA çıkarımı | 16GB+ (swap kontrol et) |

> [!WARNING]
> MPS backend'i deterministik değildir. CPU fallback planı hazır tutulmalı.

---

## FAZ 1 — Veri Altyapısı ve Determinizm

### 1.1 Deterministik Seed Altyapısı

Tüm kütüphane seed'leri tek bir config dosyasında saklanır:

```yaml
# config/seed.yaml
seeds:
  python: 42
  numpy: 42
  torch: 42
  cuda_deterministic: true
  mps_warn_only: true
```

**Ek Ayarlar:**
- `PIL.ImageFile.LOAD_TRUNCATED_IMAGES = False` — truncasyon kaynaklı non-determinizm engellenir
- `torch.set_num_threads(1)` + `OMP_NUM_THREADS=1` — multi-threading kaynaklı non-determinizm engellenir

### 1.2 Stratified 5-Fold Split

> [!IMPORTANT]
> Aynı hastanın farklı zamanlardaki görüntüleri **aynı fold'da** olmalıdır. Aksi halde data leakage oluşur.

**Adımlar:**
1. Patient ID'leri unique olarak al
2. `GroupKFold` benzeri strateji uygula (patient-level split)
3. Her fold'da minimum sınıf dağılımını kontrol et:
   - Normal: min 50 örnek
   - COVID: min 50 örnek
   - Diğer: min 50 örnek (varsa)
4. Split dosyalarını kaydet: `fold_0_train.txt`, `fold_0_val.txt`, vb.

> [!CAUTION]
> Asla kod içinde dinamik split yapma. Kayıtlı split dosyaları **source of truth** olmalıdır.

### 1.3 LMDB Cache Sistemi

**Amaç:** Her epoch'ta yapılan PIL/OpenCV resize işlemlerinin yavaşlığını ve non-determinizmini önleme.

**Hash Fonksiyonu:**
```
hash = SHA256(image_path + str(resize_size) + str(normalization_params))
```

**LMDB Oluşturma:**
1. Her görüntüyü 224×224 olarak resize et (LANCZOS veya BILINEAR — **sabit bir interpolasyon** seç)
2. Tensor → numpy array → compress (blosc/lz4)
3. Key: hash, Value: compressed array

**Validasyon:**
```python
assert torch.equal(tensor_run1, tensor_run2)  # Bitwise eşitlik şart
```

> [!TIP]
> LMDB toplam boyutu kontrol et. Çok küçükse (ör. 2GB) fazla compression veya eksik kayıt olabilir.

---

## FAZ 2 — Frozen Embedding Çıkarımı

### 2.1 TabPFN v2 Entegrasyonu (Tabular → 192-dim)

| Parametre | Değer |
|-----------|-------|
| Çıktı boyutu | 192-dim |
| Eksik değerler | `np.nan` (dummy değer koyma!) |
| Feature sırası | `feature_columns.txt`'ye kaydet |
| Checkpoint | Her 100 hastada `.npy`'ye kaydet |

**Kritik:** TabPFN'in `predict` değil `embed` fonksiyonunu kullan.

### 2.2 RadJEPA Entegrasyonu (Görüntü → 768-dim)

**MPS vs CPU Kararı:**

| Seçenek | Hız | Determinizm | Not |
|---------|-----|-------------|-----|
| CPU | 🐢 Yavaş | ✅ Garantili | Güvenli seçenek |
| MPS | 🚀 Hızlı | ⚠️ Kontrol gerekli | `torch.inference_mode()` + 3-5 örnekle CPU karşılaştırması yap |

**Post-Processing (Zorunlu):**
```python
embedding = embedding / np.linalg.norm(embedding)  # L2 Normalize
```

> [!IMPORTANT]
> RadJEPA contrastive pretraining ile eğitilmiş, cosine space'de çalışır. L2 normalizasyonu atlanmamalıdır.

**Augmentasyon:** RadJEPA frozen olduğu için **augmentasyon YAPILMAZ**.

**Cache Yapısı:**
```
/patient_001/
  ├── tabular_192.npy
  ├── radiological_768.npy
  └── metadata.json  # label, fold_idx, missing_flags
```

**Kontrol:** Embedding boyutlarını doğrula (192 ve 768). `nan`/`inf` varsa → batch size=1'e düşür.

---

## FAZ 3 — Boyut İndirgeme ve Fusion Mimarisi

### 3.1 Projection Head Tasarımı

**Parametre Bütçesi:** N=2,500 örnek → max ~250k trainable parametre (N/10 kuralı).

```
┌─────────────────┐     ┌──────────────────┐
│  Tabular Path   │     │   Vision Path    │
│  192 → 64       │     │   768 → 128      │
│  LayerNorm(64)  │     │   LayerNorm(128) │
│  GELU           │     │   GELU           │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         └───────┬───────────────┘
                 │ Concat
           ┌─────▼─────┐
           │  192-dim   │
           │ Joint Space│
           └────────────┘
```

> [!NOTE]
> **BatchNorm değil, LayerNorm** kullan — küçük batch size'da BatchNorm instabil olur.

**Kontrol:** Toplam trainable parametre ~30-40k civarı olmalı. 500k+ → overfit riski (boyutları küçült).

### 3.2 Modality Dropout

Her batch'te rastgele bir değer üret (0-1):

| Aralık | Davranış |
|--------|----------|
| `< 0.1` | Tabular = 0 vektörü, sadece görüntü |
| `0.1 - 0.2` | Görüntü = 0 vektörü, sadece tabular |
| `> 0.2` | Her iki modalite de kullanılır |

> [!IMPORTANT]
> `torch.zeros_like(input)` kullan, mean imputation **YAPMA**. LayerNorm sonrası 0 vektörü belirli bir pattern oluşturur ve model "bu modalite eksik" sinyalini öğrenir.

---

## FAZ 4 — Hiyerarşik Eğitim Protokolü (2-Stage)

### 4.1 Stage 1: Patoloji Tespiti (Binary)

| Parametre | Değer |
|-----------|-------|
| Label | Normal=0, COVID+Diğer=1 |
| Loss | `BCEWithLogitsLoss` (pos_weight) |
| Optimizer | AdamW (weight_decay=1e-4) |
| LR | 1e-3 |
| Early stopping | Val loss 10 epoch düşmezse dur |

**Class Weight:** Inverse frequency (ör. Normal=1000, Pnömoni=1500 → weights=[1.5, 1.0]).

**Kontrol:** Stage 1 sonrası val specificity > 0.95 olmalı. Değilse → data leakage olasılığı.

### 4.2 Stage 2: Diferansiyel Teşhis (COVID vs Diğer)

| Parametre | Değer |
|-----------|-------|
| Veri | Sadece Stage 1'de Pnömoni etiketli hastalar |
| Label | COVID=1, Diğer=0 |
| Class weight | Diğer Pnömoni azsa ağırlığı artır (ör. 4x) |

> [!WARNING]
> Filtreleme sonrası < 200 örnek kalırsa few-shot teknikleri düşünülmeli.

### 4.3 Inference Logic

```
Input: Hasta embedding'i (192-dim)

Step 1: Stage1_model → prob_pathology
  ├── prob_pathology < 0.5 → "Normal"  (confidence = 1 - prob_pathology)
  └── prob_pathology ≥ 0.5 →
        Step 2: Stage2_model → prob_covid
          ├── prob_covid > 0.5 → "COVID"           (conf = prob_covid × prob_pathology)
          └── prob_covid ≤ 0.5 → "Other Pneumonia" (conf = (1-prob_covid) × prob_pathology)
```

**Güven Skoru:** İki stage'in olasılıkları çarpılır (chain rule). Stage 1 düşükse Stage 2 güveni de düşer.

---

## FAZ 5 — Robustlaştırma ve Augmentasyon

### 5.1 Embedding Space Mixup

```python
lam = np.random.beta(0.2, 0.2)  # Uç değerler daha olası (%90-%10 karışımlar)
mixed_emb   = lam * emb_i + (1 - lam) * emb_j
mixed_label = lam * label_i + (1 - lam) * label_j  # Soft label
```

> [!NOTE]
> Embedding space'te mixup yapılır çünkü RadJEPA/TabPFN embedding'leri smooth manifold üzerindedir. Ham görüntüde mixup anlamsız sonuçlar üretir.

### 5.2 Gaussian Noise Regularization

```python
noise = 0.05 * batch_std * torch.randn_like(embedding)
augmented_embedding = embedding + noise
```

### 5.3 Agresif Dropout

| Alan | Dropout Oranı | Not |
|------|---------------|-----|
| Joint space (classification öncesi) | `p=0.6` | 2.5k veri için gerekli |
| Underfit durumunda | `p=0.4`'e düşür | Train loss düşmüyorsa |

---

## FAZ 6 — Kalibrasyon ve Klinik Validasyon

### 6.1 Temperature Scaling (Her Fold İçin Ayrı)

- Grid search: T = 0.5 → 3.0 (50 adım)
- Her T için ECE (Expected Calibration Error) hesapla
- En düşük ECE → seçilen T

**Stage 1 ve Stage 2 için ayrı T değerleri** (T1, T2).

### 6.2 Uncertainty Thresholding (Reject Option)

| Confidence | Çıktı |
|------------|-------|
| ≥ 0.7 | Model tahmini kabul edilir |
| < 0.7 | "Belirsiz — Radyolog İncelemesi Gerekli" |

**Hedef:** %10-15 reject oranı, kalan örneklerde %95+ accuracy.

### 6.3 Metrik Hesaplama

| Metrik | Açıklama |
|--------|----------|
| **Macro-F1** | Her sınıf için F1 → ortalama (imbalance'a duyarlı) |
| **Sensitivity@Spec=0.95** | COVID için spec=0.95 sabitken max sensitivity |
| **Calibration Curve** | Reliability diagram (predicted conf vs actual acc) |
| **Modality Ablation** | Sadece tabular / sadece görüntü / ikisi birlikte |

---

## FAZ 7 — Reprodüksiyon ve Paketleme

### 7.1 Determinizm Doğrulama
- Tüm pipeline'ı cache'siz tekrar çalıştır
- İlk vs ikinci çalıştırma metrikleri **bitwise aynı** olmalı (float32)
- Değilse → non-deterministik katman var (Dropout seed'lenmemiş, BatchNorm kullanılmış vb.)

### 7.2 Embedding Cache Checksum
- Tüm `.npy` dosyalarının SHA256 hash'leri → `manifest.json`
- Farklı makinede hash doğrulaması

### 7.3 Requirements Freeze
```bash
pip freeze > requirements.txt
```
- CUDA/MPS versiyonlarını not et

### 7.4 Dokümantasyon Çıktıları

| Dosya | İçerik |
|-------|--------|
| `fold_assignments.csv` | Hangi hasta hangi fold'da |
| `temperature_values.json` | Her fold için T1 ve T2 |
| `strategy_rationale.md` | Hiyerarşik strateji kullanılma nedeni |
| `reject_report.json` | "Belirsiz" olarak reddedilen örnekler |

---

## Sorun-Çözüm Tablosu

| Durum | Tanı | Çözüm |
|-------|------|-------|
| Val loss > train loss (sabit) | Overfitting | Dropout → 0.6, PCA (768→64), Mixup aç, epoch 50→20 |
| Sadece COVID tahmini | Class Imbalance | Stage 2 class weight 1:5, focal loss dene |
| Modality Collapse | Tabular ignore ediliyor | Modality dropout → p=0.3, tabular projection 64→128 |
| ECE > 0.2 | Kötü kalibrasyon | Grid search 0.1-5.0, Platt Scaling dene |
| MPS OOM | Memory yetmezliği | Batch size=1, RadJEPA CPU'da, gradient checkpointing |
| Aynı hasta train+val'de | Data Leakage | `GroupKFold` + patient_id bazlı split |
