"""
preprocessor.py
---------------
Text cleaning and normalization.
Uses inline stopwords (no NLTK download required) and SpaCy lemmatization.
Falls back to simple stemming if SpaCy is unavailable.
"""

import re
import string

# ---------------------------------------------------------------------------
# Inline English stopwords (no download required)
# ---------------------------------------------------------------------------
_RAW_STOPWORDS = """
i me my myself we our ours ourselves you your yours yourself yourselves
he him his himself she her hers herself it its itself they them their theirs
themselves what which who whom this that these those am is are was were be been
being have has had having do does did doing a an the and but if or because as
until while of at by for with about against between into through during before
after above below to from up down in out on off over under again further then
once here there when where why how all both each few more most other some such
no nor not only own same so than too very s t can will just don should now
ain aren couldn didn doesn hadn hasn haven isn mightn mustn needn shan
shouldn wasn weren won wouldn really very quite just now get got also
"""
STOP_WORDS = set(_RAW_STOPWORDS.split())

# ---------------------------------------------------------------------------
# Load SpaCy (optional)
# ---------------------------------------------------------------------------
def _load_spacy():
    try:
        import spacy
        return spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except Exception:
        return None

nlp = _load_spacy()

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
_HTML_RE    = re.compile(r"<[^>]+>")
_URL_RE     = re.compile(r"https?://\S+|www\.\S+")
_SPACE_RE   = re.compile(r"\s+")
_SPECIAL_RE = re.compile(r"[^a-zA-Z0-9\s]")


def _remove_noise(text: str) -> str:
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = text.lower()
    text = _SPECIAL_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text)
    return text.strip()


def _tokenize(text: str) -> list:
    """Simple whitespace tokenizer."""
    return [
        t for t in text.split()
        if t not in STOP_WORDS
        and t not in string.punctuation
        and len(t) > 2
        and not t.isdigit()
    ]


def _lemmatize_spacy(tokens: list) -> list:
    doc = nlp(" ".join(tokens))
    return [
        token.lemma_ for token in doc
        if not token.is_stop and len(token.lemma_) > 2
    ]


def _lemmatize_simple(tokens: list) -> list:
    """Rule-based suffix stripping when SpaCy unavailable."""
    result = []
    for t in tokens:
        # Very simple suffix rules
        if t.endswith("ing") and len(t) > 5:
            t = t[:-3]
        elif t.endswith("tion") and len(t) > 6:
            t = t[:-4]
        elif t.endswith("ed") and len(t) > 4:
            t = t[:-2]
        elif t.endswith("ly") and len(t) > 4:
            t = t[:-2]
        elif t.endswith("ness") and len(t) > 6:
            t = t[:-4]
        result.append(t)
    return result


def clean_text(text: str) -> str:
    """
    Full pipeline: noise removal -> tokenize -> stopword removal -> lemmatize.
    Returns a single cleaned string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = _remove_noise(text)
    tokens = _tokenize(text)
    if nlp is not None:
        tokens = _lemmatize_spacy(tokens)
    else:
        tokens = _lemmatize_simple(tokens)
    return " ".join(tokens)


def preprocess_dataframe(df, text_col="reviewText", summary_col="summary"):
    """Apply clean_text to review columns. Adds clean_text, review_length, char_length."""
    try:
        from tqdm import tqdm
        tqdm.pandas(desc="Cleaning reviews")
        apply_fn = lambda s: s.progress_apply(clean_text)
    except Exception:
        apply_fn = lambda s: s.apply(clean_text)

    df = df.copy()
    df["clean_review"]  = apply_fn(df[text_col].fillna(""))
    df["clean_summary"] = apply_fn(df[summary_col].fillna("")) if summary_col in df.columns else ""
    df["clean_text"]    = df["clean_review"] + " " + df["clean_summary"]
    df["review_length"] = df[text_col].apply(lambda x: len(str(x).split()))
    df["char_length"]   = df[text_col].apply(len)
    return df


if __name__ == "__main__":
    samples = [
        "This product is <b>AMAZING</b>! Arrived super fast. Check https://example.com",
        "Terrible quality!! Broke after 2 days. Complete waste of money.",
        "It is okay I guess. Nothing special about it.",
    ]
    for s in samples:
        print(f"Original : {s}")
        print(f"Cleaned  : {clean_text(s)}")
        print()
