# Split Validation Report

> **Date:** 2026-02-28  
> **Significance level (α):** 0.05  
> **Method:** `StratifiedGroupKFold` (patient-level, 5-fold)  
> **Label:** `destin_discharge` → Death = 1, Survived = 0

---

## 1. File Integrity

| File | Rows | Status |
|------|------|--------|
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

All 10 CSV files were successfully generated. Each fold contains approximately 3,583 training and 896 validation patients. The minor asymmetry in Fold 3 (3,584 / 895) is expected — 4,479 is not evenly divisible by 5.

---

## 2. Data Leakage Check

| Fold | train ∩ val | Status |
|------|-------------|--------|
| Fold 0 | 0 | ✅ |
| Fold 1 | 0 | ✅ |
| Fold 2 | 0 | ✅ |
| Fold 3 | 0 | ✅ |
| Fold 4 | 0 | ✅ |

No patient appears in both the training and validation sets of any fold. This guarantees zero data leakage across all splits.

---

## 3. Coverage Check

| Metric | Value | Status |
|--------|-------|--------|
| Unique patients across all val sets | 4,479 / 4,479 | ✅ |
| Duplicate patients across val sets | 0 | ✅ |

When all five validation sets are combined, every patient appears exactly once. No patient is missing or duplicated, ensuring complete and non-overlapping coverage.

---

## 4. Stratification Distribution

**Global mortality rate:** 550 Death / 3,929 Survived (12.28%)

| Fold | Train | Train Death% | Val | Val Death% | Deviation |
|------|-------|-------------|-----|-----------|-----------|
| 0 | 3,583 | 12.28% | 896 | 12.28% | 0.0000 |
| 1 | 3,583 | 12.28% | 896 | 12.28% | 0.0000 |
| 2 | 3,583 | 12.28% | 896 | 12.28% | 0.0000 |
| 3 | 3,584 | 12.28% | 895 | 12.29% | 0.0001 |
| 4 | 3,583 | 12.28% | 896 | 12.28% | 0.0000 |

**Max deviation: 0.0001 (< 2%) ✅**

The mortality rate is virtually identical across all folds. The maximum deviation from the global rate is 0.01%, confirming that `StratifiedGroupKFold` preserved the class balance near-perfectly.

---

## 5. Statistical Tests

### 5.1 Chi-Squared Homogeneity Test

> **H₀:** The mortality distribution is the same across all folds.

| Fold | Survived | Death |
|------|----------|-------|
| Val 0 | 786 | 110 |
| Val 1 | 786 | 110 |
| Val 2 | 786 | 110 |
| Val 3 | 785 | 110 |
| Val 4 | 786 | 110 |

| Metric | Value |
|--------|-------|
| χ² | 0.0001 |
| df | 4 |
| **p-value** | **1.0000** |

The chi-squared test yields p = 1.0000, meaning there is no statistically significant difference in the class distribution across folds. H₀ cannot be rejected — the distributions are homogeneous.

### 5.2 Proportion Z-Test (Each Fold vs Global)

> **H₀:** The fold's mortality rate equals the global rate (0.1228).

| Fold | Val Death% | z-stat | p-value | Status |
|------|-----------|--------|---------|--------|
| 0 | 110/896 (12.28%) | −0.0025 | 0.9980 | ✅ |
| 1 | 110/896 (12.28%) | −0.0025 | 0.9980 | ✅ |
| 2 | 110/896 (12.28%) | −0.0025 | 0.9980 | ✅ |
| 3 | 110/895 (12.29%) | +0.0100 | 0.9920 | ✅ |
| 4 | 110/896 (12.28%) | −0.0025 | 0.9980 | ✅ |

All five folds pass the individual proportion z-test with p-values exceeding 0.99. Each fold's mortality rate is statistically indistinguishable from the global rate, providing fold-level confirmation of balanced stratification.

---

## Summary

| Check | Status |
|-------|--------|
| File integrity | ✅ |
| Data leakage | ✅ |
| Coverage | ✅ |
| Stratification (< 2%) | ✅ |
| χ² homogeneity (p > 0.05) | ✅ |
| Z-test all folds (p > 0.05) | ✅ |

**All 6 checks passed.** The splits are both practically sound (no leakage, full coverage) and statistically validated (confirmed homogeneity via χ² and z-tests). Training can proceed with confidence.
