import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# 1. Define paths (using the relative path that worked for you)
INPUT_FILE = os.path.join('data', 'cve_ml_ready.csv')


def build_pipeline_skeleton():
    print("=== CVE ML Pipeline Architecture Setup ===")

    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Please run cve_ml_prep.py first.")
        return

    # 2. Load the ML-ready dataset
    print(f"Loading ML-ready dataset from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    # 3. Define Feature Columns
    # CRITICAL: We must drop 'cvss_base_score' (or similar) to prevent DATA LEAKAGE.
    # The model cannot use the score to predict the score!

    # Find the description column dynamically
    desc_col = next((col for col in df.columns if 'description' in col.lower()), None)
    # Find a categorical column (like vendor or product)
    cat_col = next((col for col in df.columns if 'vendor' in col.lower() or 'product' in col.lower()), None)

    if not desc_col:
        print("Error: Could not find a description column.")
        return

    text_features = [desc_col]
    categorical_features = [cat_col] if cat_col else []

    print(f"Text feature used: {text_features}")
    print(f"Categorical feature used: {categorical_features}")

    # 4. Branch 1: Text Processing (NLP)
    text_transformer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),  # Captures 1-word and 2-word phrases (e.g., "remote execution")
        stop_words='english'
    )

    # 5. Branch 2: Tabular Processing (Metadata)
    if categorical_features:
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    else:
        categorical_transformer = None

    # 6. Hybrid Fusion: ColumnTransformer
    transformers = [('text', text_transformer, text_features)]
    if categorical_transformer:
        transformers.append(('cat', categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'  # Automatically drops ID, target, and unused columns
    )

    # 7. Final Pipeline Skeleton
    # We define the structure now. In Week 2, we will plug in the HistGradientBoostingClassifier.
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
        # ('classifier', HistGradientBoostingClassifier(class_weight='balanced'))
    ])

    # 8. Verification Output
    print("\n✅ Pipeline architecture successfully scaffolded.")
    print(f"Pipeline steps defined: {[step[0] for step in pipeline.steps]}")
    print("Ready for model training in Week 2.")


if __name__ == "__main__":
    build_pipeline_skeleton()