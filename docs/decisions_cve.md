# Project Decisions Log (CVE Prioritization)

## Decision 1: CVE Severity Threshold Adjustment
**Author:** Aiman Mahmud  
**Student ID:** 2323081642   
**Date:** October 28, 2026

**Context:** During initial data validation (`validate_data.py`), we analyzed the class distribution of the Vinokur CVE dataset.
**Finding:** Only 7.42% of the dataset had a CVSS score $\ge$ 9.0. This falls below our agreed-upon minimum threshold of 8% for the minority class.
**Decision:** To prevent severe class imbalance and ensure the ML model has enough "Critical" examples to learn from, we adjusted the binarization threshold from `CVSS >= 9.0` to **`CVSS >= 7.0`**.
**Result:** The new class distribution is approximately 31.8% Critical (2,375 rows) and 68.2% Non-Critical (5,094 rows), providing a much healthier dataset for training.