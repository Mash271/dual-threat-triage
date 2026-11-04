import pandas as pd
import os

# 1. Define paths

INPUT_FILE = os.path.join('data', 'cve_clean.csv')
OUTPUT_FILE = os.path.join('data', 'cve_ml_ready.csv')


def prepare_ml_data():
    print("=== ML-Specific Feature Refinement ===")

    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Please run validate_data.py first.")
        return

    # 2. Load the already cleaned and binarized dataset
    print(f"Loading clean dataset from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Initial shape: {df.shape}")

    # 3. Strategic ML Decision: Drop the CWE column

    cwe_col = next((col for col in df.columns if 'cwe' in col.lower()), None)

    if cwe_col:
        df = df.drop(columns=[cwe_col])
        print(f"Dropped '{cwe_col}' column to prevent the model from memorizing CWE IDs.")
    else:
        print("No CWE column found to drop (already clean).")

    # 4. Optional: Basic NLP Text Normalization
    # Find the description column dynamically
    desc_col = next((col for col in df.columns if 'description' in col.lower()), None)
    if desc_col:
        # Convert to lowercase and strip extra whitespace for TF-IDF efficiency
        df[desc_col] = df[desc_col].astype(str).str.lower().str.strip()
        print("Normalized text descriptions to lowercase.")

    # 5. Save the final ML-ready dataset
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ ML-Ready dataset saved to: {OUTPUT_FILE}")
    print(f"Final Shape: {df.shape}")
    print(f"Target Distribution:\n{df['is_critical'].value_counts()}")


if __name__ == "__main__":
    prepare_ml_data()