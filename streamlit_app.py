"""Bartascope on Streamlit -- alternate frontend for a no-card free host.

Deliberately not a copy of app/static/index.html. Streamlit runs the app
code and the model in one process (there is no separate API to call), so
this drives app.model.Classifier directly rather than making an HTTP
request. That single-process constraint is also why app/model.py no longer
imports transformers: see CLAUDE.md's Container section and
scripts/measure_memory.py -- transformers alone cost ~385 MB RSS, which
would not have left headroom under this host's ~1 GB limit once Streamlit's
own ~70 MB baseline was added. Model + tokenizer now peaks at 427 MB.

Same behavioural rules as the FastAPI service: normalise() runs before
classification (app.model.Classifier.predict does this), the message text
is never logged, and it classifies on click only -- never on keystroke.

Streamlit Community Cloud clones this git repo, and onnx_out/ is gitignored
(build artifact, per CLAUDE.md) -- a 472 MB model has no business in git
history anyway, and Streamlit re-clones the repo on every redeploy. Resolved
by pulling the model from a public Hugging Face model repo at container
start, via BARTASCOPE_HF_REPO below, rather than Git LFS: LFS's free
bandwidth quota is 1 GB/month against a 472 MB artifact, so two redeploys in
a month would exhaust it, while HF's own CDN has no such cap on public repos.
See CLAUDE.md's Deploy section for the exact steps (creating the HF repo,
uploading onnx_out/, deploying on share.streamlit.io).

Run locally (reads onnx_out/ directly, no download):
    .venv/Scripts/streamlit run streamlit_app.py
"""

import os
import time

import streamlit as st

from app.model import Classifier, MODEL_VERSION
from app.schemas import MAX_CHARS
from app.textproc import LABELS

# Explicit path overrides win outright -- set both to skip the download
# entirely, e.g. for local testing against a directory that isn't onnx_out/.
# Otherwise: BARTASCOPE_HF_REPO set -> pull from Hugging Face; unset -> the
# local onnx_out/ default, unchanged from before this repo existed.
_MODEL_PATH_OVERRIDE = os.getenv("BARTASCOPE_MODEL_PATH")
_TOKENIZER_DIR_OVERRIDE = os.getenv("BARTASCOPE_TOKENIZER_DIR")
HF_REPO_ID = os.getenv("BARTASCOPE_HF_REPO", "")
HF_REVISION = os.getenv("BARTASCOPE_HF_REVISION", "main")
MAX_LENGTH = int(os.getenv("BARTASCOPE_MAX_LENGTH", "64"))

# model.onnx is the graph; model.onnx.data holds the weights and is loaded
# by reference from inside model.onnx, so both must land in the same
# directory. tokenizer.json is the third file app/model.py actually opens --
# tokenizer_config.json lives in the HF repo for parity with the Docker
# image but nothing here reads it.
_HF_MODEL_FILES = ("model.onnx", "model.onnx.data", "tokenizer.json")


def _resolve_model_dir() -> tuple[str, float]:
    """Return (dir containing model.onnx + tokenizer.json, download_seconds).

    download_seconds is 0.0 whenever no network fetch was needed: explicit
    path overrides, or no BARTASCOPE_HF_REPO configured (plain local dev).
    """
    if _MODEL_PATH_OVERRIDE and _TOKENIZER_DIR_OVERRIDE:
        return _TOKENIZER_DIR_OVERRIDE, 0.0
    if not HF_REPO_ID:
        return "onnx_out", 0.0

    from huggingface_hub import hf_hub_download

    t0 = time.perf_counter()
    try:
        local_dir = None
        for filename in _HF_MODEL_FILES:
            path = hf_hub_download(
                repo_id=HF_REPO_ID, filename=filename, revision=HF_REVISION,
            )
            local_dir = os.path.dirname(path)
    except Exception as exc:
        # Raised, not swallowed: main() below surfaces this via st.error and
        # st.stop(), rather than letting the app come up in a state where
        # every classification silently 500s. A missing/renamed/private repo
        # or a Hugging Face outage must be visible, not a blank result card.
        raise RuntimeError(
            f"could not download the model from the Hugging Face repo "
            f"'{HF_REPO_ID}' (revision '{HF_REVISION}'): {exc}. Check that "
            f"the repo is public and contains {', '.join(_HF_MODEL_FILES)}."
        ) from exc
    return local_dir, time.perf_counter() - t0


# Fixed at 1 rather than derived from a cgroup quota (docker-entrypoint.sh's
# approach doesn't apply here -- there is no cgroup file to read on this
# host). A single interactive visitor at a time needs no more, and an
# oversubscribed ORT thread pool spin-waits and can measure slower than a
# correctly sized one -- the same lesson documented in app/model.py, just
# with no host-reported CPU count to size it from here.
THREADS = 1

# Matches app/schemas.py's ClassifyRequest. Kept as a separate constant
# there and imported here rather than duplicated, so the two frontends can't
# quietly drift apart on the limit they enforce.
_ = MAX_CHARS

BADGE_COLORS = {
    "phishing": {"bg": "#E4D5D2", "fg": "#7A2E1C", "bar": "#A03A22"},
    "promotional": {"bg": "#F5EBD8", "fg": "#6E4B14", "bar": "#A67722"},
    "normal": {"bg": "#EDEAE4", "fg": "#4A4139", "bar": "#6B4A34"},
    "transactional": {"bg": "#EDEAE4", "fg": "#4A4139", "bar": "#6B4A34"},
}

st.set_page_config(page_title="Bartascope", page_icon="🔎", layout="centered")


@st.cache_resource(show_spinner="Loading the classifier...")
def load_classifier() -> tuple[Classifier, float]:
    # Cached as a resource, not just a value: this makes the download (if
    # any) and the ORT session creation run exactly once for the life of
    # the container, no matter how many visitors hit it. A plain module-level
    # call would do the same on the first import, but st.cache_resource is
    # what actually guarantees "once" across Streamlit's per-session reruns.
    model_dir, download_seconds = _resolve_model_dir()
    model_path = _MODEL_PATH_OVERRIDE or os.path.join(model_dir, "model.onnx")
    tokenizer_dir = _TOKENIZER_DIR_OVERRIDE or model_dir
    clf = Classifier(model_path, tokenizer_dir, MAX_LENGTH, THREADS)
    clf.load()
    return clf, download_seconds


st.title("Bartascope")
st.caption("SMS message type classifier. Advisory only.")
st.write(
    "Paste an SMS message below to see whether it looks personal, a "
    "service notification, advertising, or a phishing attempt."
)

try:
    classifier, download_seconds = load_classifier()
    load_error = None
except Exception as exc:  # noqa: BLE001 -- surfaced to the viewer, not raised
    classifier = None
    download_seconds = 0.0
    load_error = str(exc)

if load_error:
    st.error(f"The classifier could not be loaded: {load_error}")
    st.stop()

# Only meaningful the first time a given container serves a request -- every
# later rerun (this session or another visitor's) hits load_classifier()'s
# cache and gets the same download_seconds back, so this line always
# describes *this container's* cold start, not necessarily *this pageview's*.
if download_seconds:
    st.caption(
        f"Cold start: model pulled from Hugging Face in "
        f"{download_seconds:.1f}s, session created in "
        f"{classifier.load_seconds:.1f}s."
    )

if "result" not in st.session_state:
    st.session_state.result = None

message = st.text_area(
    "Message", max_chars=MAX_CHARS, height=100,
    placeholder="Your OTP is 123456. Do not share it with anyone.",
)
st.caption(f"{len(message)} / {MAX_CHARS}")

if st.button("Check message", type="primary"):
    if not message.strip():
        # Clear any previous verdict rather than leaving it on screen under
        # the warning -- a stale "normal" sitting there reads as if it
        # describes the empty input, which is the dangerous direction to be
        # wrong in on a phishing classifier. app/static/index.html hit this
        # same bug; see CLAUDE.md's Frontend section.
        st.session_state.result = None
        st.warning("Enter a message first.")
    else:
        with st.spinner("Classifying..."):
            results, inference_ms = classifier.predict([message])
        st.session_state.result = results[0] | {"inference_ms": inference_ms}

result = st.session_state.result
if result:
    colors = BADGE_COLORS[result["label"]]
    st.markdown(
        f"""
        <div style="display:flex; align-items:baseline; gap:10px; margin-top:8px;">
          <span style="background:{colors['bg']}; color:{colors['fg']};
                       padding:4px 12px; border-radius:999px;
                       font-weight:600; text-transform:capitalize;">
            {result['label']}
          </span>
          <span style="color:#5D564E;">{result['confidence'] * 100:.1f}% confidence</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Fixed order, not sorted by value, so the bars stay in the same place
    # between messages and can be compared at a glance.
    bar_rows = []
    for label in LABELS:
        prob = result["probabilities"][label]
        bar_color = BADGE_COLORS[label]["bar"]
        bar_rows.append(f"""
        <div style="display:flex; align-items:center; gap:8px; margin:4px 0;">
          <span style="width:100px; font-size:13px; text-transform:capitalize;">{label}</span>
          <div style="flex:1; background:#EDEAE4; border-radius:4px; height:10px;">
            <div style="width:{prob * 100:.1f}%; background:{bar_color};
                        height:10px; border-radius:4px;"></div>
          </div>
          <span style="width:44px; font-size:13px; text-align:right;">{prob * 100:.1f}%</span>
        </div>
        """)
    st.markdown("".join(bar_rows), unsafe_allow_html=True)

    lang_note = " (uncertain)" if result["language_uncertain"] else ""
    st.caption(
        f"Language: {result['language']}{lang_note}  ·  "
        f"{result['inference_ms']:.1f} ms  ·  {MODEL_VERSION}"
    )

st.divider()
st.markdown(
    "**Advisory only.** This is a model's best guess, not a guarantee — it "
    "can be wrong in both directions. Whatever it says, **never share an "
    "OTP, PIN or password with anyone**, and never enter one on a page you "
    "reached from a link in a message."
)
