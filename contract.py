# contract.py — do not change without both of you agreeing in writing.
RESULT_KEYS = {
    "threat_type": str,   # "PHISHING" or "CVE"
    "risk_score": float,  # 0.0 - 100.0
    "verdict": str,       # "PASS" | "WARN" | "BLOCK"
    "reasons": list,      # exactly 3 human-readable strings
    "model_version": str,
}

def validate_result(result: dict) -> None:
    """Raises AssertionError if either half breaks the contract."""
    for key, typ in RESULT_KEYS.items():
        assert key in result, f"missing key: {key}"
        assert isinstance(result[key], typ), f"{key} must be {typ.__name__}"
    assert result["threat_type"] in ("PHISHING", "CVE")
    assert result["verdict"] in ("PASS", "WARN", "BLOCK")
    assert 0.0 <= result["risk_score"] <= 100.0
    assert len(result["reasons"]) == 3