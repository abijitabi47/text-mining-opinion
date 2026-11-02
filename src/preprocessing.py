import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOP_WORDS = set(ENGLISH_STOP_WORDS) | {"product", "amazon", "item"}

def clean_text(text: str) -> str:
    """Lowercase, remove URLs/punctuation/numbers, tokenize, remove stop words, and lightly normalize tokens."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = re.findall(r"\b[a-z]{2,}\b", text)
    cleaned = []
    for token in tokens:
        if token in STOP_WORDS:
            continue
        # Lightweight suffix normalization, deliberately conservative.
        if len(token) > 5 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 5 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 4 and token.endswith("ed"):
            token = token[:-2]
        elif len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        if token and token not in STOP_WORDS:
            cleaned.append(token)
    return " ".join(cleaned)
