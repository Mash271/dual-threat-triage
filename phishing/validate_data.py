from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent
PATH = BASE / "data" / "phishing_raw.csv"
OUT  = BASE / "data" / "phishing_clean.csv"

if not PATH.exists():
    raise SystemExit(f"STOP: {PATH} not found.")

print("Loading dataset...")
try:
    df = pd.read_csv(PATH)
except UnicodeDecodeError:
    print("UTF-8 failed, retrying with latin-1...")
    df = pd.read_csv(PATH, encoding="latin-1")

df.columns = df.columns.str.strip()
df = df.rename(columns={"URL": "url"})

required = {"url", "label"}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"STOP: missing columns {missing}. Found: {list(df.columns)}")

print("\n--- Dataset Stats ---")
print("Total rows:", len(df))
print("\nClass Balance (ORIGINAL labels):")
print(df["label"].value_counts())

print("\n--- Data Quality ---")
print("Duplicate URLs:", df["url"].duplicated().sum())
print("Empty/Null URLs:", df["url"].isna().sum())
print("Empty/Null labels:", df["label"].isna().sum())

clean = df.drop_duplicates(subset="url").dropna(subset=["url", "label"])
clean = clean[clean["url"].astype(str).str.strip() != ""]

# FIX: pandas .sample() has NO stratify argument.
# Stratified sampling is done with sklearn's train_test_split instead.
if len(clean) > 30000:
    print(f"\nDataset is large ({len(clean)} rows). Downsampling to 30,000 (stratified)...")
    clean, _ = train_test_split(clean, train_size=30000, stratify=clean["label"], random_state=42)

# VISUAL CHECK: read these six URLs in the output before trusting anything.
print("\nSample URLs with ORIGINAL label == 0:")
print(clean[clean["label"] == 0]["url"].head(3).to_string())
print("\nSample URLs with ORIGINAL label == 1:")
print(clean[clean["label"] == 1]["url"].head(3).to_string())

# PhiUSIIL convention: 0 = phishing, 1 = legitimate (matches published counts).
# REMAP so that in OUR codebase 1 = phishing, forever. Recorded in decisions.md.
clean = clean.copy()
clean["label"] = (clean["label"] == 0).astype(int)

clean.to_csv(OUT, index=False)
print("\nClean rows saved to:", OUT)
print("Clean rows:", len(clean))
print("\nFinal Class Balance (AFTER remap: 1 = phishing):")
print(clean["label"].value_counts())