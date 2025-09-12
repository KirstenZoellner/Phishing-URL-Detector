from __future__ import annotations
import re
import math
import tldextract
from urllib.parse import urlparse
import numpy as np
import pandas as pd

SUSPICIOUS_TLDS = {
    "zip","mov","country","stream","download","review","click","link","work","kim","xyz","top","gq","ml","cf"
}
SHORTENERS = {"bit.ly","goo.gl","tinyurl.com","t.co","ow.ly","is.gd","buff.ly","adf.ly","cutt.ly","bit.do"}
SENSITIVE_WORDS = {"secure","account","update","verify","login","confirm","bank","free","click","signin","paypal"}

IP_PATTERN = re.compile(r"^((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)$")


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    prob = [float(s.count(c))/len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in prob)

def is_ip(hostname: str) -> bool:
    return bool(IP_PATTERN.match(hostname))

def count_chars(s: str, chars: str) -> int:
    return sum(s.count(c) for c in chars)

def extract_url_parts(url: str):
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', url):
        url = 'http://' + url
    parsed = urlparse(url)
    ext = tldextract.extract(parsed.netloc)
    subdomain = ext.subdomain or ''
    domain = ext.domain or ''
    suffix = ext.suffix or ''
    hostname = parsed.netloc
    return parsed, hostname, subdomain, domain, suffix

def extract_features_from_url(url: str) -> dict:
    try:
        parsed, hostname, subdomain, domain, suffix = extract_url_parts(url)
        full = url
        path = parsed.path or ''
        query = parsed.query or ''
        fragment = parsed.fragment or ''

        counts = {
            "count_dot": full.count('.'),
            "count_hyphen": full.count('-'),
            "count_at": full.count('@'),
            "count_question": full.count('?'),
            "count_percent": full.count('%'),
            "count_equal": full.count('='),
            "count_ampersand": full.count('&'),
            "count_slash": full.count('/'),
            "count_digit": sum(ch.isdigit() for ch in full),
            "count_letter": sum(ch.isalpha() for ch in full),
        }

        length_features = {
            "url_length": len(full),
            "hostname_length": len(hostname),
            "path_length": len(path),
            "query_length": len(query),
            "fragment_length": len(fragment),
            "subdomain_length": len(subdomain),
            "domain_length": len(domain),
            "tld_length": len(suffix),
        }

        ratio_features = {
            "digit_ratio": counts["count_digit"]/max(1, len(full)),
            "letter_ratio": counts["count_letter"]/max(1, len(full)),
            "symbol_ratio": (length_features["url_length"] - counts["count_digit"] - counts["count_letter"]) / max(1, len(full)),
        }

        subdomains = [s for s in subdomain.split('.') if s]
        subdomain_stats = {
            "num_subdomains": len(subdomains),
            "max_subdomain_length": max([len(s) for s in subdomains], default=0),
        }

        bool_int = lambda b: 1 if b else 0

        boolean_features = {
            "has_https": bool_int(parsed.scheme.lower() == "https"),
            "is_ip_in_hostname": bool_int(is_ip(hostname.split(':')[0])),
            "has_port_in_url": bool_int(':' in hostname and not is_ip(hostname)),
            "is_shortener": bool_int(hostname.lower() in SHORTENERS),
            "suspicious_tld": bool_int((suffix.lower() in SUSPICIOUS_TLDS) if suffix else 0),
            "has_sensitive_word": bool_int(any(w in full.lower() for w in SENSITIVE_WORDS)),
            "has_multiple_slashes": bool_int('//' in path.strip('/')),
        }

        # Entropie
        entropy_features = {
            "url_entropy": shannon_entropy(full),
            "hostname_entropy": shannon_entropy(hostname),
        }

        features = {**counts, **length_features, **ratio_features, **subdomain_stats, **boolean_features, **entropy_features}
        return features
    except Exception:
        # Return NaNs for failure, caller should dropna
        return None

def build_feature_frame(urls: list[str]) -> pd.DataFrame:
    rows = []
    for u in urls:
        feats = extract_features_from_url(u)
        if feats is not None:
            rows.append(feats)
        else:
            rows.append({})
    df = pd.DataFrame(rows)
    # Fill any missing with 0
    df = df.fillna(0)
    return df

def featurize(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    X = build_feature_frame(df["url"].tolist())
    feature_names = list(X.columns)
    return X, feature_names