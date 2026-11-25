import re
from urllib.parse import urlparse

# Known URL shorteners
SHORTENERS = {'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'is.gd', 'buff.ly'}

# Keywords commonly found in phishing URLs
SUSPICIOUS_KEYWORDS = {'login', 'verify', 'secure', 'account', 'bank', 'update', 'signin', 'confirm', 'password'}


def extract_features(url: str) -> dict:
    """
    Extracts 14 lexical features from a raw URL string.
    Normalizes scheme-less URLs (e.g., 'google.com') before parsing so
    domain, path, and subdomain lengths are extracted accurately.

    Args:
        url (str): Raw URL string input.

    Returns:
        dict: Dictionary of 14 numerical features ready for ML inference.
    """
    try:
        # 1. Clean whitespace and record total raw length
        url = str(url).strip()
        url_length = len(url)

        # 2. Normalize URL for urlparse:
        # If a URL omits the scheme (e.g., "google.com/search"), urlparse treats
        # the whole string as a path and leaves netloc empty. Prepending "http://"
        # forces urlparse to properly isolate the domain in netloc.
        has_scheme = bool(re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', url))
        parse_target = url if has_scheme else f"http://{url}"
        try:
            parsed = urlparse(parse_target)
        except ValueError:
            # Fallback if stray brackets trigger "Invalid IPv6 URL" in urlparse
            sanitized = parse_target.replace('[', '').replace(']', '')
            parsed = urlparse(sanitized)

        # 3. Extract URL components
        # Strip any port number (e.g., ":8080") from netloc for clean domain checks
        raw_domain = parsed.netloc.lower()
        domain = raw_domain.split(':')[0]
        path = parsed.path
        query = parsed.query

        # 4. Length features
        domain_length = len(domain)
        path_length = len(path)
        query_length = len(query)

        # 5. Character counts (evaluated on the raw input URL)
        num_dots = url.count('.')
        num_hyphens = url.count('-')
        num_at_signs = url.count('@')

        # 6. Ratios
        num_digits = sum(c.isdigit() for c in url)
        digit_ratio = num_digits / url_length if url_length > 0 else 0.0

        num_special = sum(not c.isalnum() for c in url)
        special_char_ratio = num_special / url_length if url_length > 0 else 0.0

        # 7. Boolean & heuristic features
        # Check if the host is a raw IPv4 address (e.g., 192.168.1.1)
        ip_pattern = re.compile(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$')
        has_ip_address = 1 if ip_pattern.match(domain) else 0

        # Subdomains: count dots in domain minus 1 for the main TLD
        num_subdomains = max(0, domain.count('.') - 1)

        # Check if the domain matches a known URL shortener
        uses_url_shortener = 1 if domain in SHORTENERS else 0

        # Count occurrences of common social-engineering keywords
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
            "num_subdomains": num_subdomains,
            "uses_url_shortener": uses_url_shortener,
            "suspicious_keyword_count": suspicious_keyword_count,
            "special_char_ratio": round(special_char_ratio, 4)
        }

    except Exception as exc:
        raise ValueError(f"Failed to extract features from URL '{url}': {exc}") from exc


if __name__ == "__main__":
    test_phish = "http://192.168.1.1/login-secure/verify.php?user=admin"
    test_safe_with_scheme = "https://www.google.com/search?q=cats"
    test_safe_no_scheme = "www.google.com/search?q=cats"

    print("--- Phishing Example ---")
    print(extract_features(test_phish))
    print("\n--- Safe Example (With Scheme) ---")
    print(extract_features(test_safe_with_scheme))
    print("\n--- Safe Example (Without Scheme) ---")
    print(extract_features(test_safe_no_scheme))