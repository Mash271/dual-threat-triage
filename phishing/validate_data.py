from pathlib import Path
from urllib.parse import urlparse
import pandas as pd
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent
PATH = BASE / "data" / "phishing_raw.csv"
OUT  = BASE / "data" / "phishing_clean.csv"

if not PATH.exists():
    raise SystemExit(f"STOP: {PATH} not found.")

print("Loading PhiUSIIL dataset...")
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

clean = df.drop_duplicates(subset="url").dropna(subset=["url", "label"]).copy()
clean["url"] = clean["url"].astype(str).str.strip()
clean = clean[clean["url"] != ""]

# Drop corrupted/unparseable URLs
def is_valid_url(u: str) -> bool:
    if not u.isascii():
        return False
    try:
        target = u if "://" in u else f"http://{u}"
        parsed = urlparse(target)
        return bool(parsed.netloc)
    except ValueError:
        return False

valid_mask = clean["url"].apply(is_valid_url)
print(f"Dropped {(~valid_mask).sum()} corrupted/unparseable URLs.")
clean = clean[valid_mask].copy()

# Stratified downsampling to 30,000 rows for fast, balanced training
if len(clean) > 30000:
    print(f"\nDataset is large ({len(clean)} rows). Downsampling to 30,000 (stratified)...")
    clean, _ = train_test_split(clean, train_size=30000, stratify=clean["label"], random_state=42)

# Visual check before remap
print("\nSample URLs with ORIGINAL label == 0:")
print(clean[clean["label"] == 0]["url"].head(3).to_string())
print("\nSample URLs with ORIGINAL label == 1:")
print(clean[clean["label"] == 1]["url"].head(3).to_string())

# PhiUSIIL convention: 0 = phishing, 1 = legitimate.
# Remap so that 1 = phishing, 0 = legitimate.
clean["label"] = (clean["label"] == 0).astype(int)

clean.to_csv(OUT, index=False)
print("\nClean rows saved to:", OUT)
print("Clean rows:", len(clean))
print("\nFinal Class Balance (AFTER remap: 1 = phishing):")
print(clean["label"].value_counts())