"""
SMS message type classification - dataset preparation.

Merges multiple source CSVs into one labelled corpus with uniform
normalisation, redaction, language tagging, deduplication and a
stratified split.

Usage:
    python prepare_dataset.py --outdir data/

Edit the SOURCES list below to point at your CSV files.
"""

import argparse
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# CONFIG - edit this section
# --------------------------------------------------------------------------

# One entry per input CSV.
#   path       : file location, or a list of files sharing the same schema
#   text_col   : column holding the message text
#   label_col  : column holding the original label
#   label_map  : original label (lowercased) -> unified label
#   source     : short tag recorded on every row from this file
#   lang_col   : optional column holding a language label
#   lang_map   : original language value (lowercased) -> unified language
#   lang_fixed : optional, assign this language to every row
SOURCES = [
    {
        "path": [
            "raw/bengali_sms_train.parquet",
            "raw/bengali_sms_val.parquet",
            "raw/bengali_sms_test.parquet",
        ],
        "text_col": "text",
        "label_col": "label",
        "source": "bengali_smishing",
        "label_map": {
            "normal": "normal",
            "promo": "promotional",
            "smish": "phishing",
        },
        # This dataset's column is called "source" but holds the language.
        "lang_col": "source",
        "lang_map": {
            "bengali": "bangla",
            "banglish": "banglish",
            "english": "english",
            "codemix": "codemix",
        },
    },
    {
        "path": "raw/mendeley_sms_phishing.csv",
        "text_col": "TEXT",
        "label_col": "LABEL",
        "source": "mendeley",
        "label_map": {
            "ham": "normal",
            "spam": "promotional",
            "smishing": "phishing",
        },
        "lang_fixed": "english",
    },
    {
        "path": "raw/collected_messages.csv",
        "text_col": "text",
        "label_col": "label",
        "source": "collected_phone",
        "lang_col": "lang",
        "lang_map": {l: l for l in ["bangla", "english", "banglish", "codemix"]},
        "label_map": {k: k for k in
                      ["normal", "transactional", "promotional", "phishing"]},
    },
]

LABELS = ["normal", "transactional", "promotional", "phishing"]
LANGS = ["bangla", "english", "banglish", "codemix"]

SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train, val, test
RANDOM_SEED = 42

# Rows shorter than this after normalisation are dropped as junk.
MIN_CHARS = 5

# Templated SMS ("Cash In Tk X from Y") are identical once digits are
# stripped. Keeping a few preserves template variety; keeping all of them
# lets the same template land in train and test and inflates accuracy.
# Set to 1 for aggressive dedup, or a large number to disable.
MAX_PER_TEMPLATE = 3

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

# --------------------------------------------------------------------------
# Normalisation and redaction
# --------------------------------------------------------------------------

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


def fingerprint(text: str) -> str:
    """Digit-stripped, punctuation-stripped key for near-duplicate detection.

    Templated SMS differ only in the amount or the recipient name, so we
    remove digits before comparing.
    """
    key = text.lower()
    key = re.sub(r"\d+", "", key)
    key = re.sub(r"[^\w\u0980-\u09FF]+", "", key)
    return key


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path, encoding_errors="replace")


def load_sources(sources: list) -> pd.DataFrame:
    frames = []
    for spec in sources:
        paths = spec["path"] if isinstance(spec["path"], list) else [spec["path"]]
        chunks = []
        for p in paths:
            path = Path(p)
            if not path.exists():
                print(f"  ! skipping {path} (not found)")
                continue
            chunks.append(read_any(path))
            print(f"    read {path.name}")
        if not chunks:
            continue
        df = pd.concat(chunks, ignore_index=True)

        missing = [c for c in (spec["text_col"], spec["label_col"]) if c not in df.columns]
        if missing:
            sys.exit(f"ERROR: {spec['source']} is missing column(s): {missing}\n"
                     f"       columns present: {list(df.columns)}")

        out = pd.DataFrame()
        out["text_raw"] = df[spec["text_col"]].astype(str)
        original = df[spec["label_col"]].astype(str).str.strip().str.lower()
        out["label"] = original.map(spec["label_map"])
        out["source"] = spec["source"]

        # Language: fixed value, mapped column, or left for auto-detection.
        if spec.get("lang_fixed"):
            out["lang_manual"] = spec["lang_fixed"]
        elif spec.get("lang_col") and spec["lang_col"] in df.columns:
            raw_lang = df[spec["lang_col"]].astype(str).str.strip().str.lower()
            out["lang_manual"] = raw_lang.map(spec.get("lang_map", {l: l for l in LANGS}))
            bad = raw_lang[out["lang_manual"].isna()].value_counts()
            if len(bad):
                print(f"  ! {spec['source']}: unmapped language values "
                      f"{dict(bad)} - will be auto-detected")
        else:
            out["lang_manual"] = pd.NA

        unmapped = original[out["label"].isna()].value_counts()
        if len(unmapped):
            print(f"  ! {spec['source']}: {unmapped.sum()} rows dropped, unmapped labels:")
            for lab, n in unmapped.items():
                print(f"      {lab!r}: {n}")
        out = out[out["label"].notna()]

        print(f"  loaded {len(out):>6} rows [{spec['source']}]")
        frames.append(out)

    if not frames:
        sys.exit("ERROR: no input files found. Check the SOURCES paths.")
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------

def count_matrix(df: pd.DataFrame) -> pd.DataFrame:
    m = pd.crosstab(df["label"], df["lang"])
    for lang in LANGS:
        if lang not in m.columns:
            m[lang] = 0
    for lab in LABELS:
        if lab not in m.index:
            m.loc[lab] = 0
    m = m.loc[[l for l in LABELS if l in m.index], LANGS]
    m["TOTAL"] = m.sum(axis=1)
    m.loc["TOTAL"] = m.sum(axis=0)
    return m


def source_detector(df: pd.DataFrame) -> None:
    """Can a cheap model tell which file a message came from?

    High accuracy means the sources are stylistically distinguishable, and
    the main classifier may be reading that signal instead of the message
    type. Compare against the majority-class baseline, not against 50%.
    """
    if df["source"].nunique() < 2:
        print("  only one source present - skipped")
        return

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline

    pipe = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                        min_df=3, max_features=30000, sublinear_tf=True),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    y = df["source"].to_numpy()
    scores = cross_val_score(pipe, df["text"], y, cv=5, scoring="accuracy")
    baseline = Counter(y).most_common(1)[0][1] / len(y)

    acc = scores.mean()
    print(f"  source-detector accuracy : {acc:.3f} (+/- {scores.std():.3f})")
    print(f"  majority-class baseline  : {baseline:.3f}")
    if acc > 0.95:
        print("  VERDICT: sources are trivially separable. Your classifier can")
        print("           cheat on source style. Fix normalisation, or collect")
        print("           each class from more than one source.")
    elif acc > baseline + 0.15:
        print("  VERDICT: sources are distinguishable. Watch for style leakage")
        print("           and report per-source metrics.")
    else:
        print("  VERDICT: sources look well mixed.")


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------

def stratified_split(df: pd.DataFrame, ratios: tuple, seed: int) -> pd.DataFrame:
    """Split within every (label, lang) cell, keeping template families whole.

    Templated SMS differ only in digits. If two copies of one template land
    in train and test, the test score is measuring memorisation. So the unit
    being split is the template group, not the row.

    Cells too small to divide go entirely to train - a two-row cell cannot
    populate three splits, and silently dropping it would hide the gap.
    """
    rng = np.random.default_rng(seed)
    train_r, val_r, _ = ratios
    df = df.copy()
    df["split"] = "train"

    for (label, lang), group in df.groupby(["label", "lang"], sort=False):
        n = len(group)
        if n < 10:
            print(f"  ! cell ({label}, {lang}) has {n} rows - all to train")
            continue

        # Shuffle whole template families, then fill each split in turn.
        families = [g.index.to_numpy() for _, g in group.groupby("_fp", sort=False)]
        order = rng.permutation(len(families))

        target_val = int(round(n * val_r))
        target_test = n - int(round(n * train_r)) - target_val
        n_val = n_test = 0

        for i in order:
            fam = families[i]
            if n_val < target_val:
                df.loc[fam, "split"] = "val"
                n_val += len(fam)
            elif n_test < target_test:
                df.loc[fam, "split"] = "test"
                n_test += len(fam)

    # A fingerprint can appear in more than one (label, lang) cell - usually a
    # labelling inconsistency in the source data. Cells are processed
    # independently, so force one split per fingerprint globally.
    first = df.groupby("_fp")["split"].transform("first")
    moved = (df["split"] != first).sum()
    if moved:
        print(f"  reconciled {moved} rows whose template appeared in "
              f"more than one cell")
    df["split"] = first

    return df


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="data", help="output directory")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--keep-raw", action="store_true",
                    help="keep the pre-redaction text column in the output "
                         "(off by default - it contains unredacted phone "
                         "numbers and codes)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n[1/6] loading sources")
    df = load_sources(SOURCES)
    print(f"  total {len(df)} rows")

    print("\n[2/6] normalising and redacting (uniform across all sources)")
    # Computed from the pre-redaction text, for every row from every source,
    # so the flags carry no source fingerprint.
    df["has_url"] = df["text_raw"].str.contains(RE_URL.pattern, regex=True, na=False)
    df["has_phone"] = df["text_raw"].str.contains(RE_BD_PHONE, na=False)
    df["text"] = df["text_raw"].map(normalise)
    before = len(df)
    df = df[df["text"].str.len() >= MIN_CHARS]
    print(f"  dropped {before - len(df)} rows shorter than {MIN_CHARS} chars")

    print("\n[3/6] tagging language")
    detected = df["text"].map(detect_lang)
    df["lang"] = [d[0] for d in detected]
    review = pd.Series([d[1] for d in detected], index=df.index)
    manual = df["lang_manual"].isin(LANGS)
    df.loc[manual, "lang"] = df.loc[manual, "lang_manual"]
    review.loc[manual] = False
    df["_review"] = review
    print(f"  {manual.sum()} rows used a manual language label")
    print(f"  {int(review.sum())} rows auto-detected with low confidence")

    print("\n[4/6] deduplicating")
    before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first")
    exact = before - len(df)
    df["_fp"] = df["text"].map(fingerprint)
    before2 = len(df)
    groups = df.groupby("_fp")["_fp"].transform("size")
    df = df[df["_fp"].str.len() > 0]
    df = df.groupby("_fp", sort=False, group_keys=False).head(MAX_PER_TEMPLATE)
    print(f"  removed {exact} exact duplicates")
    print(f"  removed {before2 - len(df)} near-duplicates "
          f"(same text ignoring digits, keeping up to {MAX_PER_TEMPLATE} per template)")
    biggest = groups[groups > MAX_PER_TEMPLATE].nlargest(5)
    if len(biggest):
        print(f"  largest template groups: {sorted(set(biggest.tolist()), reverse=True)[:5]}")
        print("  (inspect these - heavy templating inflates test accuracy)")
    print(f"  {len(df)} rows remain")
    df = df.drop(columns=["lang_manual"]).reset_index(drop=True)

    print("\n[5/6] diagnostics")
    print("\n  label x language counts:")
    matrix = count_matrix(df)
    for line in matrix.to_string().splitlines():
        print("    " + line)
    empty = [(l, g) for l in LABELS for g in LANGS
             if l in matrix.index and matrix.loc[l, g] < 30 and l != "TOTAL"]
    if empty:
        print("\n  cells with fewer than 30 rows (collection targets):")
        for lab, lang in empty:
            print(f"    {lab:<14} {lang:<10} {matrix.loc[lab, lang]:>5}")

    lengths = df["text"].str.split().str.len()
    print(f"\n  token length: median {lengths.median():.0f}, "
          f"p95 {lengths.quantile(0.95):.0f}, max {lengths.max():.0f}")
    print("  (use this to set max_length - do not default to 128 or 512)")

    conflict = df.groupby("_fp")["label"].nunique()
    conflicted = conflict[conflict > 1].index
    n_conf = df["_fp"].isin(conflicted).sum()
    if n_conf:
        print(f"\n  ! {len(conflicted)} templates carry conflicting labels "
              f"({n_conf} rows) - written to label_conflicts.csv")
        print("    these are annotation errors in the source datasets")

    print("\n  source-detector check:")
    source_detector(df)

    print("\n[6/6] splitting")
    df = stratified_split(df, SPLIT_RATIOS, args.seed)

    conf = df[df["_fp"].isin(conflicted)]
    if len(conf):
        conf.drop(columns=["_review"]).to_csv(outdir / "label_conflicts.csv", index=False)
    df = df.drop(columns=["_fp", "_review"])
    if not args.keep_raw:
        df = df.drop(columns=["text_raw"])
    else:
        print("  ! --keep-raw set: unredacted text is being written to disk")

    for name in ("train", "val", "test"):
        part = df[df["split"] == name]
        part.drop(columns=["split"]).to_csv(outdir / f"{name}.csv", index=False)
        print(f"  {name:<6} {len(part):>6} rows -> {outdir / f'{name}.csv'}")

    df.to_csv(outdir / "merged.csv", index=False)
    matrix.to_csv(outdir / "count_matrix.csv")
    print(f"\n  merged.csv and count_matrix.csv written to {outdir}/\n")


if __name__ == "__main__":
    main()
