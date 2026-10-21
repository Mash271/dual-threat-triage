import pandas as pd

PATH = "phishing/data/phishing_raw.csv"   # rename to whatever you downloaded
df = pd.read_csv(PATH)

# 1. Column check — stop immediately if the file isn't what we think it is
required = {"url", "label"}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"STOP: missing {missing}. Found: {list(df.columns)}")

# 2. Size and class balance — record these numbers in docs/decisions.md
print("rows:", len(df))
print(df["label"].value_counts())

# 3. Dirt check
print("duplicate URLs:", df["url"].duplicated().sum())
print("empty URLs:", df["url"].isna().sum())

# 4. Clean and save
clean = df.drop_duplicates(subset="url").dropna(subset=["url"])
clean = clean[clean["url"].astype(str).str.strip() != ""]
clean.to_csv("phishing/data/phishing_clean.csv", index=False)
print("clean rows:", len(clean))