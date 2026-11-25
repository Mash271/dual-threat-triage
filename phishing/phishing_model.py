import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from urllib.parse import urlparse

# Support both package imports (via app.py) and direct script execution
try:
    from .features import extract_features
except ImportError:
    from features import extract_features

# 1. Setup paths and load the trained brain
BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "phishing_rf_v1.pkl"

print("Loading Phishing Model...")
rf_model = joblib.load(MODEL_PATH)

# 2. Initialize the SHAP Tree Explainer
print("Initializing SHAP Explainer...")
explainer = shap.TreeExplainer(rf_model)

# Well-known high-reputation root domains (exact domain or legitimate subdomain only)
TRUSTED_ROOT_DOMAINS = {
    "google.com", "youtube.com", "github.com", "wikipedia.org",
    "microsoft.com", "apple.com", "amazon.com", "stackoverflow.com",
    "linkedin.com", "reddit.com", "mozilla.org", "python.org",
    "northsouth.edu"
}


def _extract_clean_domain(raw_url: str) -> str:
    """Safely extracts the lowercase host without port from a raw URL."""
    target = raw_url.strip()
    if "://" not in target:
        target = f"http://{target}"
    try:
        netloc = urlparse(target).netloc.lower()
    except ValueError:
        netloc = urlparse(target.replace("[", "").replace("]", "")).netloc.lower()
    return netloc.split(":")[0]


def _is_trusted_domain(domain: str) -> bool:
    """Returns True only for exact matches or true subdomains (never google.com.evil.com)."""
    return any(
        domain == root or domain.endswith(f".{root}")
        for root in TRUSTED_ROOT_DOMAINS
    )


# 3. The Contract-Compliant Prediction Function
def predict(raw_url: str) -> dict:
    """
    Takes a raw URL string, predicts if it's phishing, and explains why.
    Returns a dictionary exactly matching the team contract.
    """
    try:
        # A. Extract the features from the raw URL
        feats_dict = extract_features(raw_url)
        feature_names = list(feats_dict.keys())

        # Use a Pandas DataFrame to avoid the "feature names" warning
        X_single = pd.DataFrame([list(feats_dict.values())], columns=feature_names)

        # B. Get the prediction and probability
        proba = float(rf_model.predict_proba(X_single)[0][1])

        # Apply domain-reputation calibration for verified high-trust root domains
        domain = _extract_clean_domain(raw_url)
        trusted = _is_trusted_domain(domain)
        if trusted:
            proba = min(proba, 0.05)

        # C. Get the SHAP values for this specific URL
        shap_values = explainer.shap_values(X_single)

        # Handle both legacy list output and modern (samples, features, classes) 3D array
        if isinstance(shap_values, list):
            shap_vals_class1 = np.asarray(shap_values[1])[0]
        else:
            shap_arr = np.asarray(shap_values)
            if shap_arr.ndim == 3:
                # Shape is (1, 14, 2) -> slice sample 0, all 14 features, class 1 (Phishing)
                shap_vals_class1 = shap_arr[0, :, 1]
            else:
                shap_vals_class1 = shap_arr.flatten()

        # D. Find the top 3 features that drove this specific prediction
        feature_impact = list(zip(feature_names, shap_vals_class1))
        feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)
        top_3_features = feature_impact[:3]

        # E. Translate those features into human-readable sentences
        reasons = []
        if trusted:
            reasons.append(f"Domain '{domain}' matched verified trusted domain reputation.")
            for feat_name, impact in top_3_features[:2]:
                direction = "increased" if impact > 0 else "decreased"
                reasons.append(f"Feature '{feat_name}' {direction} the baseline structural risk.")
        else:
            for feat_name, impact in top_3_features:
                direction = "increased" if impact > 0 else "decreased"
                reasons.append(f"Feature '{feat_name}' {direction} the phishing risk.")

        # F. Determine the verdict based on calibrated threshold (0.38)
        verdict = "BLOCK" if proba >= 0.38 else "PASS"

        # G. Return the exact contract dictionary
        return {
            "threat_type": "PHISHING",
            "risk_score": round(float(proba * 100), 2),  # Convert 0.0-1.0 to 0-100
            "verdict": verdict,
            "reasons": reasons,
            "model_version": "phishing_rf_v1"
        }
    except Exception as exc:
        raise RuntimeError(f"Prediction failed for URL '{raw_url}': {exc}") from exc


# --- Quick Test ---
if __name__ == "__main__":
    test_urls = [
        "http://192.168.1.1/login-secure/verify.php?user=admin",
        "https://www.google.com/search?q=cats&ie=UTF-8",
        "https://www.google.com"
    ]
    for test_url in test_urls:
        result = predict(test_url)
        print("\n--- Prediction Result ---")
        print(f"URL: {test_url}")
        for key, value in result.items():
            print(f"{key}: {value}")