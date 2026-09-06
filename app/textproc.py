"""
Text normalisation and language tagging, shared by the dataset builder and
the serving path.

This module exists so there is exactly one definition of `normalise`. The
model was fine-tuned on normalised text: URLs reduced to scheme+domain,
phone numbers replaced with <PHONE>, OTP digits with <CODE>. A request that
skips this step hands the model a distribution it never saw during training,
and the failure is silent -- plausible probabilities, quietly worse.

Deliberately free of pandas, numpy and torch, so the runtime image can
import it without pulling in the training stack.

Note what normalisation does NOT do: it keeps URLs, digits and punctuation
rather than stripping them. Those are the strongest phishing signals in the
data, and standard tutorial cleaning would destroy them.
"""

import re
import unicodedata

LABELS = ["normal", "transactional", "promotional", "phishing"]
LANGS = ["bangla", "english", "banglish", "codemix"]

BANGLA_BLOCK = re.compile(r"[\u0980-\u09FF]")
LATIN_LETTER = re.compile(r"[A-Za-z]")
BANGLA_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
RE_URL = re.compile(r"\b((?:https?://|www\.)[^\s]+)", re.IGNORECASE)
RE_BD_PHONE = re.compile(r"(?:\+?88)?0\d{9,10}\b")
RE_LONG_DIGITS = re.compile(r"\b\d{8,}\b")
RE_OTP = re.compile(
    r"\b(otp|pin|code|password|passcode|verification)\b[^0-9\n]{0,20}(\d{4,8})\b",
    re.IGNORECASE,
)

# Romanised Bangla function words. Used only to separate `banglish` from
# `english` in Latin-script messages. Deliberately common, short words that
# rarely appear in English SMS.
BANGLISH_MARKERS = {
    "apnar", "apni", "ache", "achen", "hoye", "hoyeche", "korun", "kore",
    "korte", "jonno", "ekhon", "bhai", "apu", "taka", "amar", "ami", "tomar",
    "kono", "khub", "valo", "bhalo", "nai", "hobe", "diye", "theke", "jabe",
    "korben", "dhonnobad", "onek", "shudhu", "jodi", "cholbe", "lagbe",
    "pathan", "pathiye", "gese", "gece", "chai", "sesh", "bondho", "notun",
}


def mask_url(match: re.Match) -> str:
    """Keep scheme and domain, drop path and query.

    The domain carries the phishing signal (bkash-verify-bd.info); the path
    can carry per-recipient tracking tokens, so it goes.
    """
    url = match.group(1)
    stripped = re.sub(r"^https?://", "", url, flags=re.IGNORECASE)
    domain = re.split(r"[/?#]", stripped, maxsplit=1)[0]
    return f"http://{domain}/<PATH>" if len(stripped) > len(domain) else f"http://{domain}"


def normalise(text: str) -> str:
    """Applied identically to every row from every source.

    Uniform treatment matters: if only the collected messages carry <PHONE>
    placeholders, that token becomes a perfect source predictor.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200c", "").replace("\u200d", "")  # ZWNJ / ZWJ
    text = text.translate(BANGLA_DIGITS)

    text = RE_EMAIL.sub("<EMAIL>", text)
    text = RE_URL.sub(mask_url, text)
    text = RE_OTP.sub(lambda m: f"{m.group(1)} <CODE>", text)
    text = RE_BD_PHONE.sub("<PHONE>", text)
    text = RE_LONG_DIGITS.sub("<NUM>", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_lang(text: str) -> tuple:
    """Return (lang, needs_review).

    Script ratio separates bangla / codemix / latin. Splitting latin into
    english vs banglish is a word-list heuristic and is flagged for review.

    The model never sees this value. It is reported to the caller as
    context, and it is a heuristic, not a language identifier.
    """
    bangla = len(BANGLA_BLOCK.findall(text))
    latin = len(LATIN_LETTER.findall(text))
    total = bangla + latin
    if total == 0:
        return "english", True

    ratio = bangla / total
    if ratio >= 0.85:
        return "bangla", False
    if ratio >= 0.05:
        return "codemix", False

    tokens = set(re.findall(r"[a-z]+", text.lower()))
    hits = len(tokens & BANGLISH_MARKERS)
    if hits >= 2:
        return "banglish", False
    if hits == 1:
        return "banglish", True
    return "english", False
