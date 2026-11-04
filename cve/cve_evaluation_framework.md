# CVE Triage AI: Mathematical Evaluation Framework

**Author:** Aiman Mahmud  
**Student ID:** 2323081642  
**Date:** October 28, 2026  

## 1. Class Imbalance Analysis
Analysis of the raw CVE dataset reveals a severe class imbalance:
- **Total Samples:** 7,469
- **Critical CVEs (CVSS ≥ 9.0):** 554 (7.42%)
- **Imbalance Ratio:** ~12.5 : 1

Because the minority class (Critical) is so small, a naive model that simply guesses "Non-Critical" for every input would achieve 92.58% accuracy while failing to identify a single critical threat. This is known as the **Accuracy Paradox**.

## 2. Selected Evaluation Metrics
To address the Accuracy Paradox, we will strictly avoid standard Accuracy and use the following metrics:

### 2.1 Precision-Recall Area Under Curve (PR-AUC)
- **Formula:** Area under the curve plotting Precision ($TP / (TP + FP)$) against Recall ($TP / (TP + FN)$) at various thresholds.
- **Justification:** Unlike ROC-AUC, PR-AUC focuses entirely on the performance of the minority (positive) class. It heavily penalizes the model for False Positives, which is crucial when the negative class (Non-Critical) is massive.

### 2.2 F2-Score
- **Formula:** $F_2 = 5 \times \frac{Precision \times Recall}{(4 \times Precision) + Recall}$
- **Justification:** The F2-score weights Recall twice as heavily as Precision. In cybersecurity, a False Negative (missing a critical CVE) is vastly more dangerous than a False Positive (flagging a safe CVE). Therefore, we optimize for Recall.

## 3. Decision Threshold Logic
The model will output a probability score $P \in [0, 1]$. We will map this to a 3-tier alert system:
- **CRITICAL:** $P \ge 0.85$ (Immediate patching required)
- **WARNING:** $0.60 \le P < 0.85$ (Schedule review)
- **LOW:** $P < 0.60$ (Ignore)