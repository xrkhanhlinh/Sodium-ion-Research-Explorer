"""Conservative text normalization and auditable, weak domain labels."""
import re
import unicodedata
from functools import lru_cache

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

SIB = re.compile(r"\b(?:sodium[\s-]*ion|Na[\s-]*ion)\s+(?:batter(?:y|ies)|cells?)\b", re.I)
SODIUM = re.compile(r"\bsodium\b|\bNa(?:\+|[\s-]*ions?\b)|\bNa[x\d][A-Za-z\d().+-]*", re.I)
ELECTRO = re.compile(r"batter|electrod|cathod|anod|electrolyt|electrochem|intercalat|deintercalat|energy storage|ion storage", re.I)
STORAGE = re.compile(r"sodium[\s-]*(?:ion[\s-]*)?(?:storage|intercalat|insertion)|(?:storage|intercalat|insertion)\w*\s+(?:of\s+)?(?:sodium|Na[\s-]*ion)|Na[\s-]*ion\s+storage", re.I)
SENSOR = re.compile(r"sweat|sodium[\s-]*(?:sens|detect)|sodium[\s-]*selective|ion[\s-]*selective\s+(?:electrod|sensor)|desalinat|blood|urine", re.I)
OTHER_BATTERY = re.compile(r"(?:lithium|potassium|zinc|aluminum|aluminium|magnesium|calcium)[\s-]*(?:ion[\s-]*)?batter", re.I)
OTHER_SODIUM = re.compile(r"sodium[\s-]*(?:metal[\s-]*chloride|sulphur|sulfur|air)\s+batter|\bZEBRA\b", re.I)
FORMULA = re.compile(r"(?<![A-Za-z])(?:[A-Z][a-z]?(?:\d+(?:\.\d+)?|[xyzδ])?(?:\([^\s()]{1,24}\)\d*)?){1,14}(?:[+-])?(?![A-Za-z])")
TOKEN = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")
LEMMATIZER = WordNetLemmatizer()


def normalize_text(value):
    if value is None:
        return ""
    text = unicodedata.normalize("NFC", str(value))
    # Convert Unicode subscripts without changing superscripts/charges or case.
    text = text.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\u00ad]", "", text)
    text = "".join(c if not unicodedata.category(c).startswith("C") or c.isspace() else " " for c in text)
    return re.sub(r"\s+", " ", text).strip()


def fingerprint(text):
    return normalize_text(text).casefold()


def relevance(title, abstract):
    """Rule labels are evidence for review, never expert ground truth."""
    text = normalize_text(title + " " + abstract)
    title_sib = bool(SIB.search(title))
    contextual_sib = bool(SIB.search(text)) or bool(re.search(r"\bSIBs?\b", text) and ELECTRO.search(text))
    if title_sib:
        return "CORE", "explicit_sodium_ion_battery_context"
    if OTHER_SODIUM.search(title):
        return "UNCERTAIN", "alternative_sodium_battery_chemistry_requires_review"
    if contextual_sib:
        if SENSOR.search(title) or (OTHER_BATTERY.search(title) and not SODIUM.search(title)):
            return "UNCERTAIN", "SIB_mentioned_but_primary_title_application_differs"
        return "CORE", "explicit_sodium_ion_battery_context"
    if OTHER_SODIUM.search(text):
        return "UNCERTAIN", "alternative_sodium_battery_chemistry_requires_review"
    if SENSOR.search(title) and not (STORAGE.search(text) and ELECTRO.search(title)):
        return "OUT_OF_SCOPE", "primary_sensing_or_biological_application_without_SIB_evidence"
    if OTHER_BATTERY.search(title) and not SODIUM.search(text):
        return "OUT_OF_SCOPE", "other_battery_system_without_sodium_evidence"
    if SODIUM.search(text) and ELECTRO.search(text):
        if STORAGE.search(text):
            return "RELATED", "sodium_storage_or_intercalation_with_electrochemical_context"
        if re.search(r"\b(?:NaxCoO2|Na\d*(?:\.\d+)?CoO2)\b", text, re.I):
            return "RELATED", "sodium_cobalt_oxide_with_electrochemical_context"
        # Sodium metal plating alone is not sufficient evidence for Na-ion storage.
        return "UNCERTAIN", "sodium_and_electrochemistry_but_system_not_established"
    return "UNCERTAIN", "insufficient_sodium_ion_battery_evidence"


def formula_spans(text):
    out = []
    for match in FORMULA.finditer(text):
        token = match.group()
        if re.search(r"\d|[()+-]", token) and re.match(r"[A-Z]", token):
            out.append(match)
        elif token.startswith("Na") and len(token) > 2 and re.search(r"[A-Z]", token[2:]):
            out.append(match)
    return out


@lru_cache(maxsize=100000)
def lemma(word):
    noun = LEMMATIZER.lemmatize(word, "n")
    return LEMMATIZER.lemmatize(noun, "v")


def lda_tokens(text, phrases=(), extra_stopwords=()):
    text = normalize_text(text)
    protected = {}
    spans = formula_spans(text)
    for i, match in reversed(list(enumerate(spans))):
        # Alphabetic placeholders survive tokenization but are never lemmatized.
        marker = "chemplaceholder" + chr(97 + i // 26) + chr(97 + i % 26)
        protected[marker] = match.group()
        text = text[:match.start()] + " " + marker + " " + text[match.end():]
    text = text.lower()
    phrase_map = {}
    for i, phrase in enumerate(sorted(phrases, key=len, reverse=True)):
        marker = "phraseplaceholder" + chr(97 + i)
        pattern = r"\b" + r"[\s-]+".join(map(re.escape, phrase.split())) + r"\b"
        text = re.sub(pattern, marker, text)
        phrase_map[marker] = phrase.replace(" ", "_")
    stops = set(ENGLISH_STOP_WORDS) | set(extra_stopwords)
    out = []
    for word in TOKEN.findall(text):
        if word in protected:
            out.append(protected[word])
        elif word in phrase_map:
            out.append(phrase_map[word])
        elif any(marker in word for marker in phrase_map):
            # TOKEN retains hyphenated compounds (e.g. all-phraseplaceholderi).
            # Restore their protected phrase before lemmatization as well.
            restored = re.sub(r"phraseplaceholder[a-z]", lambda m: phrase_map[m.group()], word)
            out.append(restored)
        elif len(word) > 1 and word not in stops and not word.startswith("chemplaceholder"):
            out.append(lemma(word))
    return out
