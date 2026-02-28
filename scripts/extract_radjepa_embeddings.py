"""
RadJEPA — Radyolojik Embedding Çıkarımı (768-dim + L2 Norm).

Pipeline:
    xray.lmdb → RadJEPA (frozen, CPU) → 768-dim embedding → L2 normalize
    → embeddings.h5 (radiological grubu)

Kullanım:
    PYTHONPATH=. uv run python scripts/extract_radjepa_embeddings.py
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import lmdb
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.config import get_config
from src.data.embedding_cache import (
    EmbeddingCacheWriter,
    RADIOLOGICAL_DIM,
    cache_stats,
)
from src.utils import set_seeds

# ── Config ───────────────────────────────────────────────────────────────
_cfg = get_config()

LMDB_PATH = _cfg.paths.xray_lmdb
H5_PATH = _cfg.paths.embeddings_h5
SEED = _cfg.seed

# RadJEPA model ID (HuggingFace)
MODEL_ID = "AIDElab-IITBombay/RadJEPA"

# ImageNet normalization (RadJEPA pretrained)
IMAGENET_MEAN = list(_cfg.image.imagenet_mean)
IMAGENET_STD = list(_cfg.image.imagenet_std)
IMAGE_SIZE = _cfg.image.size


def get_transform() -> transforms.Compose:
    """RadJEPA inference transform — augmentasyon yok."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_radjepa_model() -> torch.nn.Module:
    """RadJEPA modelini HuggingFace'ten yükle (frozen, CPU)."""
    from transformers import AutoModel

    print(f"  📦 RadJEPA modeli yükleniyor: {MODEL_ID}")
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval()

    # MPS varsa kullan (M2), yoksa CPU fallback
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"  🚀 Device: MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print(f"  🐢 Device: CPU")
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  📐 Parametre: {n_params:,}")

    return model, device


def get_lmdb_keys(lmdb_path: Path) -> list[str]:
    """LMDB'deki tüm image key'lerini döndür."""
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, subdir=True)
    with env.begin() as txn:
        all_keys: list[str] = json.loads(txn.get(b"__keys__").decode())
    env.close()
    return sorted(all_keys)


def parse_patient_info(lmdb_key: str) -> tuple[int, int]:
    """LMDB key'inden patient_id ve xray_idx çıkar.

    Key format: pXXXXXXXX/sXXXXXXXX/XXXXXXXX_XXXX
    Aynı patient'a ait birden fazla key varsa xray_idx artar.

    Returns:
        (patient_id, xray_idx) — xray_idx dışarıda hesaplanır.
    """
    pid_str = lmdb_key.split("/")[0]  # "p10030053"
    patient_id = int(pid_str.lstrip("p"))
    return patient_id, 0  # xray_idx dışarıda sayılacak


def main() -> None:
    print("=" * 60)
    print("  Sprint 2.3 — RadJEPA Embedding Çıkarımı (768-dim)")
    print("=" * 60 + "\n")

    # 1) Seed
    set_seeds()
    print(f"🌱 Seed: {SEED}\n")

    # 2) Model yükle
    print("🧠 Model yükleniyor...")
    model, device = load_radjepa_model()
    transform = get_transform()
    print()

    # 3) LMDB key'lerini al
    print(f"📂 LMDB okunuyor: {LMDB_PATH}")
    all_keys = get_lmdb_keys(LMDB_PATH)
    print(f"   Toplam X-ray: {len(all_keys):,}\n")

    # Her patient için xray_idx sayacı
    patient_xray_counter: dict[int, int] = {}

    # 4) LMDB aç ve embedding çıkar
    print(f"🩻 Embedding çıkarımı başlıyor ({device}, batch=1)...")
    env = lmdb.open(str(LMDB_PATH), readonly=True, lock=False, subdir=True,
                    readahead=False, meminit=False)

    n_success = 0
    n_error = 0

    with EmbeddingCacheWriter(H5_PATH) as writer:
        with torch.inference_mode():
            for key in tqdm(all_keys, desc="RadJEPA", unit="img"):
                try:
                    # Patient info
                    patient_id, _ = parse_patient_info(key)
                    xray_idx = patient_xray_counter.get(patient_id, 0)
                    patient_xray_counter[patient_id] = xray_idx + 1

                    # LMDB'den image oku
                    with env.begin() as txn:
                        raw_bytes = txn.get(key.encode("utf-8"))

                    if raw_bytes is None:
                        print(f"  ⚠️  Key bulunamadı: {key}")
                        n_error += 1
                        continue

                    # PIL → transform → tensor
                    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                    tensor = transform(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

                    # Model forward
                    output = model(tensor)

                    # Output handling — model çıktısına göre embedding al
                    if hasattr(output, "last_hidden_state"):
                        # HuggingFace standard output
                        # Global average pooling over spatial dims
                        emb = output.last_hidden_state.mean(dim=1)  # (1, D)
                    elif hasattr(output, "pooler_output"):
                        emb = output.pooler_output  # (1, D)
                    elif isinstance(output, torch.Tensor):
                        emb = output
                        if emb.dim() == 3:
                            emb = emb.mean(dim=1)  # (1, seq_len, D) → (1, D)
                    else:
                        # Tuple çıktı
                        emb = output[0]
                        if emb.dim() == 3:
                            emb = emb.mean(dim=1)

                    emb = emb.squeeze(0).cpu().numpy().astype(np.float32)  # (D,)

                    # L2 normalize (RadJEPA cosine space)
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm

                    # Boyut kontrolü
                    if emb.shape[0] != RADIOLOGICAL_DIM:
                        print(
                            f"  ⚠️  Boyut uyumsuz: beklenen {RADIOLOGICAL_DIM}, "
                            f"alınan {emb.shape[0]} (key={key})"
                        )
                        n_error += 1
                        continue

                    # HDF5'e yaz
                    writer.save_radiological(patient_id, emb, xray_idx=xray_idx)
                    n_success += 1

                except Exception as e:
                    print(f"  ❌ Hata (key={key}): {e}")
                    n_error += 1
                    continue

    env.close()

    # 5) Sonuçlar
    unique_patients = len(patient_xray_counter)
    print(f"\n   ✅ Başarılı: {n_success:,} embedding")
    print(f"   ❌ Hata    : {n_error}")
    print(f"   👥 Unique hasta: {unique_patients:,}")
    print(f"   🩻 Ortalama X-ray/hasta: {n_success / max(unique_patients, 1):.1f}")

    # 6) L2 norm doğrulama (ilk 5 embedding)
    print(f"\n🔍 L2 Norm Doğrulama (ilk 5):")
    import h5py
    with h5py.File(H5_PATH, "r") as f:
        if "radiological" in f:
            keys = sorted(f["radiological"].keys())[:5]
            for k in keys:
                emb = f["radiological"][k][:]
                norm = np.linalg.norm(emb)
                print(f"   {k}: norm={norm:.6f}, shape={emb.shape}")

    # 7) Cache stats
    stats = cache_stats(H5_PATH)
    print(f"\n📊 Cache İstatistikleri:")
    print(f"   Radiological embeddings : {stats['num_radiological_embeddings']}")
    print(f"   Tabular embeddings      : {stats['num_tabular_embeddings_total']}")
    print(f"   Unique patients (rad.)  : {stats['num_unique_patients_radiological']}")

    print("\n" + "=" * 60)
    print("  🎉 RadJEPA embedding çıkarımı tamamlandı!")
    print("=" * 60)


if __name__ == "__main__":
    main()
