"""One-off memory measurement for the Streamlit feasibility question.

Not part of the serving path. Answers a single question: does loading the
deployed fp32 ONNX graph plus its tokenizer fit inside a ~1 GB Streamlit
Community Cloud container, given that the model would share a process with
the Streamlit server itself (unlike the FastAPI service, which is its own
process)?

Uses the `tokenizers` package directly (Tokenizer.from_file) rather than
transformers.AutoTokenizer -- transformers' import alone cost ~385 MB in the
first measurement, dwarfing the model. scripts/verify_tokenizer_parity.py
confirmed this produces identical input_ids/attention_mask across all 200
test messages before this swap was made.

Reports peak working set (Windows' analog of Linux peak RSS) at each stage,
in-process, via psutil -- checked at each stage rather than only at the end,
so a memory spike that is later freed still shows up. Run standalone:

    .venv/Scripts/python.exe scripts/measure_memory.py
"""

import gc
import os
import sys
import time

import psutil

MODEL_PATH = "onnx_out/model.onnx"
TOKENIZER_JSON = "onnx_out/tokenizer.json"
MAX_LENGTH = 64

SAMPLE_MESSAGES = [
    "Your OTP is 482913. Do not share this code with anyone.",
    "Hey are we still on for dinner tonight? Let me know",
]

proc = psutil.Process(os.getpid())


def snapshot(label: str) -> None:
    gc.collect()
    mem = proc.memory_info()
    rss_mb = mem.rss / 1e6
    peak_mb = getattr(mem, "peak_wset", mem.rss) / 1e6
    print(f"{label:<32} rss={rss_mb:8.1f} MB   peak_wset={peak_mb:8.1f} MB")


def main() -> None:
    snapshot("process start")

    import numpy as np  # noqa: F401
    snapshot("after import numpy")

    import onnxruntime as ort
    snapshot("after import onnxruntime")

    from tokenizers import Tokenizer
    snapshot("after import tokenizers")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    t0 = time.perf_counter()
    session = ort.InferenceSession(
        MODEL_PATH, opts, providers=["CPUExecutionProvider"])
    load_s = time.perf_counter() - t0
    snapshot(f"after ORT session load ({load_s:.2f}s)")

    tokenizer = Tokenizer.from_file(TOKENIZER_JSON)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<pad>"),
                              pad_token="<pad>")
    snapshot("after tokenizer load")

    input_names = [i.name for i in session.get_inputs()]
    encodings = tokenizer.encode_batch(SAMPLE_MESSAGES)
    enc = {
        "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
        "attention_mask": np.array([e.attention_mask for e in encodings],
                                    dtype=np.int64),
    }
    feed = {n: enc[n] for n in input_names}
    session.run(["logits"], feed)
    snapshot("after first inference")

    # A few more, in case ORT grows scratch buffers past the first call.
    for _ in range(10):
        session.run(["logits"], feed)
    snapshot("after 10 more inferences")

    final = proc.memory_info()
    print()
    print(f"FINAL peak working set: {getattr(final, 'peak_wset', final.rss) / 1e6:.1f} MB")


if __name__ == "__main__":
    sys.exit(main())
