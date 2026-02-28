# 🏥 Multimodal Mortalite Tahmini Pipeline'ı

> **Proje**: CDSL Göğüs Röntgeni + Tabular Veri ile **Binary Mortalite Sınıflandırma (Death vs Survived)**  
> **Mimari**: Frozen RadJEPA (Görüntü) + Frozen TabPFN v2 (Tabular) → Projection Head → Binary Classifier  
> **Hedef**: Deterministik, reprodüse edilebilir, klinik güvenilirlikte bir mortalite tahmin sistemi

---

## İçindekiler

1. [FAZ 0 — Pre-Flight Check](#faz-0--pre-flight-check)
2. [FAZ 1 — Veri Altyapısı ve Determinizm](#faz-1--veri-altyapısı-ve-determinizm)
3. [FAZ 2 — Frozen Embedding Çıkarımı](#faz-2--frozen-embedding-çıkarımı)
4. [FAZ 3 — Boyut İndirgeme ve Fusion Mimarisi](#faz-3--boyut-indirgeme-ve-fusion-mimarisi)
5. [FAZ 4 — Binary Eğitim Protokolü](#faz-4--binary-eğitim-protokolü)
6. [FAZ 5 — Robustlaştırma ve Augmentasyon](#faz-5--robustlaştırma-ve-augmentasyon)
7. [FAZ 6 — Kalibrasyon ve Klinik Validasyon](#faz-6--kalibrasyon-ve-klinik-validasyon)
8. [FAZ 7 — Reprodüksiyon ve Paketleme](#faz-7--reprodüksiyon-ve-paketleme)
9. [Sorun-Çözüm Tablosu](#sorun-çözüm-tablosu)

---

## FAZ 0 — Pre-Flight Check

### 0.1 Exploratory Data Analysis (EDA)

Veri setine dokunmadan önce sınıf dağılımını analiz et.

**Binary Sınıflandırma (Death vs Survived):**

| Sınıf | Label | Kaynak Sütun | Açıklama |
|-------|-------|-------------|----------|
| Death | 1 | `destin_discharge == "Death"` | Hastanede ölen hastalar (550) |
| Survived | 0 | `destin_discharge != "Death"` | Taburcu olan hastalar (3,929) |

> [!NOTE]
> **Karar:** Tek aşamalı binary classification: mortalite tahmini. `destin_discharge` sütunundaki "Death" değeri pozitif sınıf olarak kullanılır. Diğer tüm taburculuk durumları (Home, Transfer, Voluntary Discharge, vb.) "Survived" olarak etiketlenir.

**Eksik Veri Analizi:**
- `glu_first_emerg` (%99 NaN) → silindi
- Temiz veri: 4,479 satır × 17 sütun (`tabpfn_features_clean.csv`)
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

Tüm kütüphane seed'leri merkezi config dosyasında saklanır:

```yaml
# config/config.yaml
seed: 42
cv:
  n_folds: 10
```

**Ek Ayarlar:**
- `PIL.ImageFile.LOAD_TRUNCATED_IMAGES = False` — truncasyon kaynaklı non-determinizm engellenir
- `torch.set_num_threads(1)` + `OMP_NUM_THREADS=1` — multi-threading kaynaklı non-determinizm engellenir

### 1.2 Stratified 10-Fold Split

> [!IMPORTANT]
> Aynı hastanın farklı zamanlardaki görüntüleri **aynı fold'da** olmalıdır. Aksi halde data leakage oluşur.

**Adımlar:**
1. Patient ID'leri unique olarak al
2. `StratifiedKFold(n_splits=10)` uygula (patient-level, mortalite label'ına göre strateji)
3. Her fold'da minimum sınıf dağılımını kontrol et
4. Split dosyalarını kaydet: `fold_{0-9}_train.csv`, `fold_{0-9}_val.csv`

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

> [!NOTE]
> Embedding çıkarımı **train'den ÖNCE** bir kez yapılır ve HDF5'e cache'lenir. Eğitim sırasında tekrar çıkarım yok.
>
> **Çalıştırma Sırası:**
> ```bash
> # 1) TabPFN (per-fold, ~10 ayrı model fit)
> PYTHONPATH=. uv run python scripts/extract_tabpfn_embeddings.py
>
> # 2) RadJEPA (tüm veri, tek seferlik)
> PYTHONPATH=. uv run python scripts/extract_radjepa_embeddings.py
> ```

### 2.1 TabPFN v2 Entegrasyonu (Tabular → 192-dim)

| Parametre | Değer |
|-----------|-------|
| Çıktı boyutu | 192-dim |
| Eksik değerler | `np.nan` (dummy değer koyma!) |
| Feature sırası | `feature_columns.txt`'ye kaydet |
| Checkpoint | Her 100 hastada `.npy`'ye kaydet |

**Kritik:** TabPFN'in `predict` değil `embed` fonksiyonunu kullan.

> [!CAUTION]
> **Hybrid Per-Fold Cache (Data Leakage Önlemi):**
> TabPFN, RadJEPA’dan farklı olarak **label bilgisini görerek** embedding üretir.
> Tüm veriyi tek seferde TabPFN’e verip sonra fold’lara bölmek **data leakage** oluşturur.
>
> **Çözüm:** Her fold `k` için ayrı bir TabPFN modeli sadece fold `k`’nın **train** setindeki
> verilerle eğitilir, sonra fold `k`’nın **val** setindeki hastalara embedding üretir.
> Böylece hiçbir validasyon hastasının label’ı, kendi embedding’ini üreten modele sızmaz.
>
> **HDF5 Yapısı:** `tabular/fold_{k}/p{pid}` (RadJEPA fold-agnostic kalır)

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
embeddings.h5
├── radiological/
│   ├── p10030053_0    # fold-agnostic (RadJEPA label görmez)
│   └── ...
├── tabular/
│   ├── fold_0/
│   │   ├── p10394712  # Fold 0 val hastası, Fold 0 train ile eğitilmiş model
│   │   └── ...
│   ├── fold_1/
│   │   └── ...
│   └── fold_{n}/
└── metadata/
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

## FAZ 4 — Binary Eğitim Protokolü

### 4.1 Tek Aşamalı Binary Training (Death vs Survived)

| Parametre | Değer |
|-----------|-------|
| Label | Survived=0, Death=1 |
| Loss | `BCEWithLogitsLoss` (pos_weight ile class dengeleme) |
| Optimizer | AdamW (weight_decay=1e-4) |
| LR | 1e-3 |
| Early stopping | Val loss 10 epoch düşmezse dur |
| Cross-validation | 10-fold (patient-level StratifiedKFold) |

**Class Weight:** Inverse frequency hesapla:
```python
pos_weight = n_survived / n_death  # ≈ 7.14 (3,929 / 550)
```

**Kontrol:** Val AUROC > 0.80 ve Sensitivity@Specificity=0.95 takip edilmeli. Mortalite sınıf dengesizliği yüksek (~7:1), Focal Loss alternatif olarak düşünülebilir.

### 4.2 Inference Logic

```
Input: Hasta embedding'i (192-dim fused)

Model → logit → sigmoid → prob_death
  ├── prob_death > 0.5 → "Death"      (confidence = prob_death)
  └── prob_death ≤ 0.5 → "Survived"   (confidence = 1 - prob_death)
```

**Güven Skoru:** Sigmoid çıktısı direkt confidence olarak kullanılır. Kalibrasyon sonrası Temperature Scaling ile düzeltilir.

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

**Tek T değeri** (binary classifier için).

### 6.2 Uncertainty Thresholding (Reject Option)

| Confidence | Çıktı |
|------------|-------|
| ≥ 0.7 | Model tahmini kabul edilir |
| < 0.7 | "Belirsiz — Radyolog İncelemesi Gerekli" |

**Hedef:** %10-15 reject oranı, kalan örneklerde %95+ accuracy.

### 6.3 Metrik Hesaplama

| Metrik | Açıklama |
|--------|----------|
| **AUROC** | Binary classifier için birincil metrik |
| **F1-Score** | Precision/Recall dengesi |
| **Sensitivity@Spec=0.95** | Mortalite için spec=0.95 sabitken max sensitivity |
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
| `temperature_values.json` | Her fold için T değeri |
| `strategy_rationale.md` | Binary classification seçim nedeni |
| `reject_report.json` | "Belirsiz" olarak reddedilen örnekler |

---

## Sorun-Çözüm Tablosu

| Durum | Tanı | Çözüm |
|-------|------|-------|
| Val loss > train loss (sabit) | Overfitting | Dropout → 0.6, PCA (768→64), Mixup aç, epoch 50→20 |
| Sadece Survived tahmini (Death hiç tahmin edilmiyor) | Şiddetli Class Imbalance (~7:1) | `pos_weight≈7.14` ayarla, focal loss dene, oversampling |
| Modality Collapse | Tabular ignore ediliyor | Modality dropout → p=0.3, tabular projection 64→128 |
| ECE > 0.2 | Kötü kalibrasyon | Grid search 0.1-5.0, Platt Scaling dene |
| MPS OOM | Memory yetmezliği | Batch size=1, RadJEPA CPU'da, gradient checkpointing |
| Aynı hasta train+val'de | Data Leakage | `GroupKFold` + patient_id bazlı split |
