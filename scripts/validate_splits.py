from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = PROJECT_ROOT / "data" / "splits" / "5fold"
CLEAN_CSV = PROJECT_ROOT / "data" / "processed" / "tabpfn_features_clean.csv"
REPORT_DIR = PROJECT_ROOT / "docs"
REPORT_PATH = REPORT_DIR / "split_validation_report.md"

N_FOLDS = 5
PASS = "✅"
FAIL = "❌"
ALPHA = 0.05  # anlamlılık düzeyi


def proportions_ztest(count: int, nobs: int, p0: float) -> tuple[float, float]:
    """Tek örneklem oran z-testi (H0: p = p0)."""
    p_hat = count / nobs
    se = np.sqrt(p0 * (1 - p0) / nobs)
    z = (p_hat - p0) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))  # two-sided
    return z, p_value


def main() -> None:
    lines: list[str] = []  # markdown rapor satırları

    def log(text: str = "") -> None:
        print(text)
        lines.append(text)

    log("# 📊 Split Validation Report")
    log("")
    log(f"> **Tarih:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    log(f"> **Anlamlılık düzeyi (α):** {ALPHA}")
    log("")
    log("---")
    log("")

    # ── 1. Dosya bütünlüğü ──────────────────────────────────────
    log("## 1. Dosya Bütünlüğü")
    log("")
    log("| Dosya | Satır | Durum |")
    log("|-------|-------|-------|")

    all_files_ok = True
    for fold in range(N_FOLDS):
        for split in ("train", "val"):
            p = SPLITS_DIR / f"fold_{fold}_{split}.csv"
            exists = p.exists()
            n = len(pd.read_csv(p)) if exists else 0
            status = PASS if exists and n > 0 else FAIL
            log(f"| `fold_{fold}_{split}.csv` | {n:,} | {status} |")
            if not exists or n == 0:
                all_files_ok = False
    log("")

    if not all_files_ok:
        log(f"**{FAIL} Eksik/boş dosyalar var, diğer kontroller atlanıyor.**")
        _save_report(lines)
        return

    # ── Veri yükleme ────────────────────────────────────────────
    clean = pd.read_csv(CLEAN_CSV)
    all_patient_ids = set(clean["patient_id"].values)
    total_patients = len(all_patient_ids)

    folds: dict[int, dict[str, pd.DataFrame]] = {}
    for fold in range(N_FOLDS):
        folds[fold] = {
            "train": pd.read_csv(SPLITS_DIR / f"fold_{fold}_train.csv"),
            "val": pd.read_csv(SPLITS_DIR / f"fold_{fold}_val.csv"),
        }

    # ── 2. Data leakage ─────────────────────────────────────────
    log("## 2. Data Leakage Kontrolü")
    log("")
    log("| Fold | train ∩ val | Durum |")
    log("|------|-------------|-------|")

    leakage_ok = True
    for fold in range(N_FOLDS):
        train_ids = set(folds[fold]["train"]["patient_id"])
        val_ids = set(folds[fold]["val"]["patient_id"])
        overlap = len(train_ids & val_ids)
        status = PASS if overlap == 0 else FAIL
        log(f"| Fold {fold} | {overlap} | {status} |")
        if overlap > 0:
            leakage_ok = False
    log("")

    # ── 3. Kapsam ───────────────────────────────────────────────
    log("## 3. Kapsam Kontrolü")
    log("")

    all_val_ids: list[int] = []
    for fold in range(N_FOLDS):
        all_val_ids.extend(folds[fold]["val"]["patient_id"].tolist())

    unique_val = set(all_val_ids)
    dup_count = len(all_val_ids) - len(unique_val)
    coverage_ok = unique_val == all_patient_ids

    log(f"- Val'de unique hasta: **{len(unique_val):,}** / {total_patients:,} "
        f"{PASS if len(unique_val) == total_patients else FAIL}")
    log(f"- Tekrarlanan hasta (val'ler arası): **{dup_count}** "
        f"{PASS if dup_count == 0 else FAIL}")
    log("")

    # ── 4. Stratifikasyon ───────────────────────────────────────
    log("## 4. Stratifikasyon Dağılımı")
    log("")

    all_labels = pd.concat([folds[0]["train"], folds[0]["val"]])
    global_death = int((all_labels["label"] == 1).sum())
    global_survived = int((all_labels["label"] == 0).sum())
    global_rate = global_death / len(all_labels)

    log(f"**Global:** {global_death} Death / {global_survived} Survived "
        f"(oran: {global_rate:.4f})")
    log("")
    log("| Fold | Train | Train Death% | Val | Val Death% | Sapma |")
    log("|------|-------|-------------|-----|-----------|-------|")

    max_deviation = 0.0
    for fold in range(N_FOLDS):
        t = folds[fold]["train"]
        v = folds[fold]["val"]
        t_rate = t["label"].mean()
        v_rate = v["label"].mean()
        dev = abs(v_rate - global_rate)
        max_deviation = max(max_deviation, dev)
        log(f"| {fold} | {len(t):,} | {t_rate:.2%} | {len(v):,} | {v_rate:.2%} | {dev:.4f} |")

    strat_ok = max_deviation < 0.02
    log("")
    log(f"Max sapma: **{max_deviation:.4f}** "
        f"{'(< 2%)' if strat_ok else '(≥ 2% ⚠️)'} {PASS if strat_ok else FAIL}")
    log("")

    # ── 5. İstatistiksel Testler ────────────────────────────────
    log("## 5. İstatistiksel Testler")
    log("")

    # 5a. Chi-squared: fold'lar arası homojenlik
    log("### 5.1 Chi-Squared Homojenlik Testi")
    log("")
    log("> **H₀:** Tüm fold'lardaki mortalite oranları aynı dağılımdan geliyor")
    log("")

    contingency = np.zeros((N_FOLDS, 2), dtype=int)
    for fold in range(N_FOLDS):
        v = folds[fold]["val"]
        contingency[fold, 0] = int((v["label"] == 0).sum())  # survived
        contingency[fold, 1] = int((v["label"] == 1).sum())  # death

    chi2, p_chi, dof, _ = stats.chi2_contingency(contingency)
    chi_ok = p_chi > ALPHA

    log("| Fold | Survived | Death |")
    log("|------|----------|-------|")
    for fold in range(N_FOLDS):
        log(f"| Val {fold} | {contingency[fold, 0]} | {contingency[fold, 1]} |")
    log("")
    log(f"| Metrik | Değer |")
    log(f"|--------|-------|")
    log(f"| χ² | {chi2:.4f} |")
    log(f"| df | {dof} |")
    log(f"| **p-value** | **{p_chi:.6f}** |")
    log(f"| Sonuç | {PASS + ' H₀ reddedilemez → dağılımlar homojen' if chi_ok else FAIL + ' H₀ reddedildi → dağılımlar farklı ⚠️'} |")
    log("")

    # 5b. Proportion z-test: her fold vs global
    log("### 5.2 Proportion Z-Test (Her Fold vs Global)")
    log("")
    log(f"> **H₀:** Fold'un mortalite oranı = global oran ({global_rate:.4f})")
    log("")
    log("| Fold | Val Death% | z-stat | p-value | Sonuç |")
    log("|------|-----------|--------|---------|-------|")

    all_z_ok = True
    for fold in range(N_FOLDS):
        v = folds[fold]["val"]
        n_death = int((v["label"] == 1).sum())
        n_total = len(v)
        z_stat, p_val = proportions_ztest(n_death, n_total, global_rate)
        ok = p_val > ALPHA
        if not ok:
            all_z_ok = False
        log(f"| {fold} | {n_death}/{n_total} ({n_death/n_total:.2%}) "
            f"| {z_stat:+.4f} | {p_val:.6f} | {PASS if ok else FAIL} |")
    log("")

    if all_z_ok:
        log(f"{PASS} **Tüm fold'lar global oranla istatistiksel olarak uyumlu (p > {ALPHA})**")
    else:
        log(f"{FAIL} **Bazı fold'lar global orandan anlamlı sapma gösteriyor**")
    log("")

    # ── Genel Sonuç ─────────────────────────────────────────────
    log("---")
    log("")
    log("## Sonuç")
    log("")
    all_ok = all_files_ok and leakage_ok and coverage_ok and strat_ok and chi_ok and all_z_ok

    log("| Kontrol | Durum |")
    log("|---------|-------|")
    log(f"| Dosya bütünlüğü | {PASS if all_files_ok else FAIL} |")
    log(f"| Data leakage | {PASS if leakage_ok else FAIL} |")
    log(f"| Kapsam | {PASS if coverage_ok else FAIL} |")
    log(f"| Stratifikasyon (< 2%) | {PASS if strat_ok else FAIL} |")
    log(f"| χ² homojenlik (p > {ALPHA}) | {PASS if chi_ok else FAIL} |")
    log(f"| Z-test tüm fold'lar (p > {ALPHA}) | {PASS if all_z_ok else FAIL} |")
    log("")
    log(f"**{'✅ TÜM KONTROLLER GEÇTİ' if all_ok else '❌ BAZI KONTROLLER BAŞARISIZ'}**")

    _save_report(lines)


def _save_report(lines: list[str]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n📄 Rapor kaydedildi → {REPORT_PATH}")


if __name__ == "__main__":
    main()
