"""Verifies that loading tokenizer.json via the raw `tokenizers` package
produces byte-identical input_ids/attention_mask to AutoTokenizer, before
app/model.py is allowed to switch off transformers.

transformers' fast tokenizer *is* a `tokenizers.Tokenizer` under the hood, but
AutoTokenizer additionally applies tokenizer_config.json settings (e.g.
add_prefix_space, max_length) that a bare `Tokenizer.from_file` does not pick
up automatically -- truncation and the special-token template are baked into
tokenizer.json itself (verified by inspection: truncation.max_length=64,
post_processor is a TemplateProcessing adding <s>/</s>), but padding is not
(padding: null in the file), so it must be enabled explicitly to match
padding="longest".

Checks both single-message encoding (batch size 1, exercises truncation and
special tokens with no padding) and one big batch of all 200 test messages
(exercises dynamic padding to the batch's longest sequence).

Usage: .venv/Scripts/python.exe scripts/verify_tokenizer_parity.py
"""

import sys

import numpy as np
import pandas as pd
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from app.textproc import normalise

TOKENIZER_DIR = "onnx_out"
TOKENIZER_JSON = "onnx_out/tokenizer.json"
MAX_LENGTH = 64
N_SAMPLES = 200


def load_messages() -> list:
    df = pd.read_csv("data/merged.csv")
    texts = df[df.split == "test"].text.astype(str).tolist()[:N_SAMPLES]
    return [normalise(t) for t in texts]


def encode_old_batch(tok, texts):
    enc = tok(texts, return_tensors="np", padding="longest",
              truncation=True, max_length=MAX_LENGTH)
    return enc["input_ids"], enc["attention_mask"]


def encode_new_batch(tok, texts):
    tok.enable_padding(pad_id=tok.token_to_id("<pad>"), pad_token="<pad>")
    encodings = tok.encode_batch(texts)
    ids = np.array([e.ids for e in encodings], dtype=np.int64)
    mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
    return ids, mask


def compare(label, old_ids, old_mask, new_ids, new_mask) -> bool:
    ok = old_ids.shape == new_ids.shape and old_mask.shape == new_mask.shape \
        and np.array_equal(old_ids, new_ids) \
        and np.array_equal(old_mask, new_mask)
    status = "MATCH" if ok else "MISMATCH"
    print(f"[{status}] {label}: old shape={old_ids.shape} new shape={new_ids.shape}")
    if not ok:
        n = min(old_ids.shape[0] if old_ids.ndim else 1,
                new_ids.shape[0] if new_ids.ndim else 1)
        for i in range(n):
            o, nn = np.atleast_2d(old_ids)[i], np.atleast_2d(new_ids)[i]
            if o.shape != nn.shape or not np.array_equal(o, nn):
                print(f"  first diff at row {i}:")
                print(f"    old input_ids: {o.tolist()}")
                print(f"    new input_ids: {nn.tolist()}")
                break
    return ok


def main() -> int:
    old_tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    new_tok = Tokenizer.from_file(TOKENIZER_JSON)

    texts = load_messages()
    print(f"loaded {len(texts)} normalised test messages")

    all_ok = True

    # Batch size 1, one at a time -- exercises truncation + special tokens,
    # no padding ambiguity possible.
    single_mismatches = 0
    for text in texts:
        o_ids, o_mask = encode_old_batch(old_tok, [text])
        n_ids, n_mask = encode_new_batch(new_tok, [text])
        if not (np.array_equal(o_ids, n_ids) and np.array_equal(o_mask, n_mask)):
            single_mismatches += 1
            if single_mismatches <= 3:
                print(f"MISMATCH (single) on: {text!r}")
                print(f"  old: {o_ids.tolist()}")
                print(f"  new: {n_ids.tolist()}")
    if single_mismatches:
        print(f"[MISMATCH] {single_mismatches}/{len(texts)} single-message encodings differ")
        all_ok = False
    else:
        print(f"[MATCH] all {len(texts)} single-message encodings identical")

    # One batch of all 200 -- exercises dynamic padding to a shared longest
    # sequence.
    o_ids, o_mask = encode_old_batch(old_tok, texts)
    n_ids, n_mask = encode_new_batch(new_tok, texts)
    all_ok &= compare("full batch (dynamic padding)", o_ids, o_mask, n_ids, n_mask)

    print()
    print("RESULT:", "PASS -- identical" if all_ok else "FAIL -- see mismatches above")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
