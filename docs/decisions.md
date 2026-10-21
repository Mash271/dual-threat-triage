## [October 22nd, 2026] — Task F1 Complete (Phishing Data & Features)

### Dataset Discovery & Train/Serve Skew Decision
- **Dataset:** PhiUSIIL mirror from Kaggle (235,795 raw rows).
- **Feature Discovery:** The dataset ships with ~50 pre-computed features. Some are string-derivable (URLLength, IsHTTPS), but others are *page-content* features (HasFavicon, NoOfPopup, LineOfCode) which require visiting and rendering the webpage.
- **Critical Decision (Train/Serve Skew):** We will **NOT** use the page-content features for our primary model. Our Streamlit app scores URLs pasted by users in real-time without fetching them (fetching is slow, unsafe, and fails on dead links). If we trained on page-content features, the model would break at inference time. We will strictly use our 14 custom lexical extractors computed from the raw URL string to ensure zero train/serve skew. 
- *(Stretch goal: Train a secondary model on all 50 features offline just to report an "upper bound" comparison in the final report).*

### Data Cleaning & Balancing
- **Label Remap:** Original dataset convention was 0 = Phishing, 1 = Legitimate. Visual sample check PASSED. Remapped in `phishing_clean.csv` so that **1 = Phishing, 0 = Legitimate** for our codebase.
- **Downsampling:** Stratified downsample from 235,370 clean rows to 30,000 rows (seed 42) via `sklearn.train_test_split` to keep CPU training fast.
- **Final Balance:** 12,812 Phishing (43%) / 17,188 Legitimate (57%). 
- **Decision:** Balance is healthy. No `class_weight` adjustments needed for W5 training.
- **Task F1 Status:** COMPLETE. `validate_data.py` and `features.py` are functional.