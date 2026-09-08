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

NOT YET WIRED UP: Streamlit Community Cloud clones this git repo, and
onnx_out/ is gitignored (build artifact, per CLAUDE.md). Getting the 472 MB
model into the deployed environment -- committed via Git LFS vs. pulled from
a separate Hugging Face model repo at cold start -- is an open question with
its own cold-start and quota tradeoffs, not decided here. Locally this reads
onnx_out/ directly and works today.

Run locally:
    .venv/Scripts/streamlit run streamlit_app.py
"""

import os

import streamlit as st

from app.model import Classifier, MODEL_VERSION
from app.schemas import MAX_CHARS
from app.textproc import LABELS

MODEL_PATH = os.getenv("BARTASCOPE_MODEL_PATH", "onnx_out/model.onnx")
TOKENIZER_DIR = os.getenv("BARTASCOPE_TOKENIZER_DIR", "onnx_out")
MAX_LENGTH = int(os.getenv("BARTASCOPE_MAX_LENGTH", "64"))

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
def load_classifier() -> Classifier:
    clf = Classifier(MODEL_PATH, TOKENIZER_DIR, MAX_LENGTH, THREADS)
    clf.load()
    return clf


st.title("Bartascope")
st.caption("SMS message type classifier. Advisory only.")
st.write(
    "Paste an SMS message below to see whether it looks personal, a "
    "service notification, advertising, or a phishing attempt."
)

try:
    classifier = load_classifier()
    load_error = None
except Exception as exc:  # noqa: BLE001 -- surfaced to the viewer, not raised
    classifier = None
    load_error = str(exc)

if load_error:
    st.error(f"The classifier could not be loaded: {load_error}")
    st.stop()

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
