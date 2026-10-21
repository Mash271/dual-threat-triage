import re
from urllib.parse import urlparse

# Known URL shorteners
SHORTENERS = {'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'is.gd', 'buff.ly'}
# Keywords commonly found in phishing URLs
SUSPICIOUS_KEYWORDS = {'login', 'verify', 'secure', 'account', 'bank', 'update', 'signin', 'confirm', 'password'}

def extract_features(url: str) -> dict:
    """
    Extracts 14 lexical features from a raw URL string.
    Returns a dictionary of numerical features ready for ML.
    """
    url = str(url).strip()
    url_length = len(url)
    
    # Parse URL into components
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path
    query = parsed.query
    
    # Length features
    domain_length = len(domain)
    path_length = len(path)
    query_length = len(query)
    
    # Character counts
    num_dots = url.count('.')
    num_hyphens = url.count('-')
    num_at_signs = url.count('@')
    
    # Ratios
    num_digits = sum(c.isdigit() for c in url)
    digit_ratio = num_digits / url_length if url_length > 0 else 0
    
    num_special = sum(not c.isalnum() for c in url)
    special_char_ratio = num_special / url_length if url_length > 0 else 0
    
    # Boolean / Complex features
    # 1. IP Address check (matches standard IPv4 format)
    ip_pattern = re.compile(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$')
    has_ip_address = 1 if ip_pattern.match(domain) else 0
    
    # 2. HTTPS check
    has_https = 1 if parsed.scheme == 'https' else 0
    
    # 3. Subdomains (dots in domain minus 1 for the main TLD)
    num_subdomains = max(0, domain.count('.') - 1)
    
    # 4. Shortener check
    uses_url_shortener = 1 if domain in SHORTENERS else 0
    
    # 5. Suspicious keyword count
    url_lower = url.lower()
    suspicious_keyword_count = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url_lower)

    return {
        "url_length": url_length,
        "domain_length": domain_length,
        "path_length": path_length,
        "query_length": query_length,
        "num_dots": num_dots,
        "num_hyphens": num_hyphens,
        "num_at_signs": num_at_signs,
        "digit_ratio": round(digit_ratio, 4),
        "has_ip_address": has_ip_address,
        "has_https": has_https,
        "num_subdomains": num_subdomains,
        "uses_url_shortener": uses_url_shortener,
        "suspicious_keyword_count": suspicious_keyword_count,
        "special_char_ratio": round(special_char_ratio, 4)
    }

# Quick test to ensure it works
if __name__ == "__main__":
    test_phish = "http://192.168.1.1/login-secure/verify.php?user=admin"
    test_safe = "https://www.google.com/search?q=cats"
    
    print("--- Phishing Example ---")
    print(extract_features(test_phish))
    print("\n--- Safe Example ---")
    print(extract_features(test_safe))