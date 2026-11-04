# %% [markdown]
# # 1. False Negative Inspector
# Let's find the 29 phishing URLs that fooled our Random Forest and see why.

# %%
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from features import extract_features

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "data" / "phishing_clean.csv")
rf = joblib.load(BASE / "models" / "phishing_rf_v1.pkl")

# Recreate the EXACT same test split (seed 42)
X_urls = df["url"]
y = df["label"]
_, X_te_urls, _, y_te = train_test_split(X_urls, y, test_size=0.2, stratify=y, random_state=42)

# Extract features for the test set
print("Extracting features for test set...")
X_te_feats = pd.DataFrame(X_te_urls.apply(extract_features).tolist())

# Predict
y_pred = rf.predict(X_te_feats)

# Find False Negatives (Actual = 1 [Phishing], Predicted = 0 [Legit])
fn_mask = (y_te == 1) & (y_pred == 0)
fn_urls = X_te_urls[fn_mask].values
fn_feats = X_te_feats[fn_mask]

print(f"\nFound {len(fn_urls)} False Negatives. Here are the first 5:\n")
for i in range(min(5, len(fn_urls))):
    print(f"URL: {fn_urls[i]}")
    print(f"Features: https={fn_feats.iloc[i]['has_https']}, length={fn_feats.iloc[i]['url_length']}, digits={fn_feats.iloc[i]['digit_ratio']:.2f}\n")


# %% [markdown]
# # 2. The 50-Feature Upper Bound (Offline Experiment)
# What if we used ALL the pre-computed columns from the dataset? 
# This shows what we traded away for real-time deployability.

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score

# Drop the raw text and the target label
X_full = df.drop(columns=["url", "label"])

# CRITICAL FIX: Scikit-learn RF cannot handle raw strings (like 'Domain' or 'TLD').
# We must filter to keep ONLY numeric columns.
X_full = X_full.select_dtypes(include=[np.number])

X_tr, X_te, y_tr, y_te = train_test_split(X_full, df["label"], test_size=0.2, stratify=df["label"], random_state=42)

print(f"Training on {X_full.shape[1]} pre-computed numeric features...")
rf_full = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_full.fit(X_tr, y_tr)

preds = rf_full.predict(X_te)
print("\n--- 50-Feature Model Results ---")
print(f"Accuracy: {accuracy_score(y_te, preds):.4f} (vs 0.9942 for 14-feature)")
print(f"Recall:   {recall_score(y_te, preds):.4f} (vs 0.9887 for 14-feature)")
print(f"F1 Score: {f1_score(y_te, preds):.4f} (vs 0.9931 for 14-feature)")


# %% [markdown]
# # 3. Threshold Analysis & Precision-Recall Curve
# At what confidence score does our model become nearly perfect at flagging phishing?

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve

# Get probabilities (confidence scores between 0.0 and 1.0)
probas = rf.predict_proba(X_te_feats)[:, 1]

# Calculate PR curve
precision, recall, thresholds = precision_recall_curve(y_te, probas)

# Plot the curve
plt.figure(figsize=(8, 5))
plt.plot(recall, precision, marker='.', color='b', linewidth=2)
plt.xlabel('Recall (How many phishing sites we catch)', fontsize=12)
plt.ylabel('Precision (How accurate our "Phishing" flags are)', fontsize=12)
plt.title('Precision-Recall Curve (RandomForest)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.show()

# Find the "Sweet Spot" thresholds
# 1. Max precision achieved
max_prec_idx = np.argmax(precision[:-1])
print(f"Max Precision: {precision[max_prec_idx]*100:.2f}% at threshold {thresholds[max_prec_idx]:.3f} (Recall: {recall[max_prec_idx]*100:.2f}%)")

# 2. Threshold for 99.5% Precision (if it exists)
target = 0.995
valid = np.where(precision[:-1] >= target)[0]
if len(valid) > 0:
    idx = valid[0]
    print(f"To get >= {target*100}% Precision: Set threshold to {thresholds[idx]:.3f} (Recall drops to {recall[idx]*100:.2f}%)")
else:
    print(f"Model does not reach {target*100}% precision on this test set.")