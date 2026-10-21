import pandas as pd

df = pd.read_csv("cve/data/cve_raw.csv")

# Kaggle files name these differently — find them, don't guess
score_col = next((c for c in df.columns if "cvss" in c.lower()), None)
desc_col  = next((c for c in df.columns if "description" in c.lower()), None)
if score_col is None or desc_col is None:
    raise SystemExit(f"STOP: need cvss + description columns. Found: {list(df.columns)}")

df = df.dropna(subset=[desc_col, score_col])
df = df[df[desc_col].astype(str).str.strip() != ""]

crit = (df[score_col] >= 9.0).mean()
print("rows:", len(df), "| critical fraction at >=9.0:", round(crit, 4))

if crit < 0.08:   # the fallback rule we agreed on
    print("DECISION: binarizing at CVSS >= 7.0 — record in docs/decisions.md")
    df["is_critical"] = (df[score_col] >= 7.0).astype(int)
else:
    df["is_critical"] = (df[score_col] >= 9.0).astype(int)

print(df["is_critical"].value_counts())
df.to_csv("cve/data/cve_clean.csv", index=False)