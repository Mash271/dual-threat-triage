import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from features import extract_features

# 1. Setup paths and load the trained brain
BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "phishing_rf_v1.pkl"

print("Loading Phishing Model...")
rf_model = joblib.load(MODEL_PATH)

# 2. Initialize the SHAP Tree Explainer
print("Initializing SHAP Explainer...")
explainer = shap.TreeExplainer(rf_model)

# 3. The Contract-Compliant Prediction Function
def predict(raw_url: str) -> dict:
    """
    Takes a raw URL string, predicts if it's phishing, and explains why.
    Returns a dictionary exactly matching the team contract.
    """
    # A. Extract the 14 features from the raw URL
    feats_dict = extract_features(raw_url)
    feature_names = list(feats_dict.keys())
    
    # FIX 1: Use a Pandas DataFrame to avoid the "feature names" warning
    X_single = pd.DataFrame([list(feats_dict.values())], columns=feature_names)
    
    # B. Get the prediction and probability
    prediction = rf_model.predict(X_single)[0]
    proba = rf_model.predict_proba(X_single)[0][1] # Probability of class 1 (Phishing)
    
    # C. Get the SHAP values for this specific URL
    shap_values = explainer.shap_values(X_single)
    
    # FIX 2: Handle SHAP output shapes safely
    if isinstance(shap_values, list):
        # Binary classification returns a list of 2 arrays. We want class 1 (Phishing).
        shap_vals_class1 = shap_values[1]
    else:
        shap_vals_class1 = shap_values
        
    # Flatten to ensure it's a 1D array of 14 numbers, not a 2D grid
    shap_vals_class1 = np.asarray(shap_vals_class1).flatten()
    
    # D. Find the top 3 features that drove this specific prediction
    feature_impact = list(zip(feature_names, shap_vals_class1))
    feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)
    top_3_features = feature_impact[:3]
    
    # E. Translate those features into human-readable sentences
    reasons = []
    for feat_name, impact in top_3_features:
        direction = "increased" if impact > 0 else "decreased"
        reasons.append(f"Feature '{feat_name}' {direction} the phishing risk.")
        
    # F. Determine the verdict based on probability (Default 0.5 threshold for now)
    verdict = "BLOCK" if proba >= 0.5 else "PASS"
    
    # G. Return the exact contract dictionary
    return {
        "threat_type": "PHISHING",
        "risk_score": round(float(proba * 100), 2), # Convert 0.0-1.0 to 0-100
        "verdict": verdict,
        "reasons": reasons,
        "model_version": "phishing_rf_v1"
    }

# --- Quick Test ---
if __name__ == "__main__":
    test_url = "http://192.168.1.1/login-secure/verify.php?user=admin"
    result = predict(test_url)
    
    print("\n--- Prediction Result ---")
    print(f"URL: {test_url}")  # <--- ADD THIS LINE
    for key, value in result.items():
        print(f"{key}: {value}")