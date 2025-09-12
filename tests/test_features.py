from src.features import extract_features_from_url, featurize
import pandas as pd

def test_feature_extraction_basic():
    url = "https://www.example.com/login?user=abc#frag"
    feats = extract_features_from_url(url)
    assert feats is not None
    assert feats["has_https"] == 1
    assert feats["url_length"] > 0
    assert feats["count_question"] == 1

def test_featurize_frame():
    df = pd.DataFrame({"url": ["example.com", "http://1.2.3.4:8080/a-b"]})
    X, names = featurize(df)
    assert X.shape[0] == 2
    assert len(names) == X.shape[1]