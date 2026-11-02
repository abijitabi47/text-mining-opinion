import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

RELATED_TAGS = {
    "battery": ["battery life", "charging"],
    "sound": ["audio quality", "volume"],
    "price": ["value for money", "affordability"],
    "comfort": ["fit", "usability"],
    "quality": ["build quality", "durability"],
    "delivery": ["shipping", "packaging"],
    "camera": ["photo quality", "video quality"],
    "performance": ["speed", "reliability"],
}

def generate_tags(texts, top_n=40):
    """Rank tags by mean TF-IDF, refine simple duplicates, and add transparent rule-based related tags."""
    texts = [str(t) for t in texts if str(t).strip()]
    if not texts:
        return pd.DataFrame(columns=["tag", "score", "stage"])
    vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 2), min_df=1)
    X = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()
    scores = X.mean(axis=0).A1
    raw = pd.DataFrame({"tag": terms, "score": scores}).sort_values("score", ascending=False)
    # Refine: keep one of obvious morphological variants and remove tags that are substrings of a stronger tag.
    kept = []
    for row in raw.itertuples():
        tag = row.tag.strip()
        if any(tag == existing or tag in existing.split() and len(tag) < 5 for existing in kept):
            continue
        kept.append(tag)
        if len(kept) >= top_n:
            break
    refined = raw[raw["tag"].isin(kept)].copy()
    refined["stage"] = "ranked/refined"
    present = " ".join(texts).lower()
    enriched = []
    for keyword, additions in RELATED_TAGS.items():
        if keyword in present:
            for addition in additions:
                if addition not in set(refined["tag"]):
                    enriched.append({"tag": addition, "score": 0.0, "stage": f"enriched from '{keyword}'"})
    result = pd.concat([refined, pd.DataFrame(enriched)], ignore_index=True)
    return result.sort_values(["score", "tag"], ascending=[False, True]).reset_index(drop=True)
