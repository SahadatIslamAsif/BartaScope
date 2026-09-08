"""Measures a bare Streamlit server's own memory footprint, with no model
loaded, so it can be added to scripts/measure_memory.py's number. In a
Streamlit app the server and the app code share one process -- there is no
separating them at request time the way the FastAPI service separates HTTP
handling from inference -- so this baseline is additive, not a separate
budget.

Launches `streamlit run` on a trivial one-line app, polls until it serves,
reads the child process's peak working set, then kills it.

    .venv/Scripts/python.exe scripts/measure_streamlit_baseline.py
"""

import subprocess
import sys
import tempfile
import time
import urllib.request

import psutil

PORT = 8765

TRIVIAL_APP = "import streamlit as st\nst.write('hello')\n"


def wait_for_server(port: int, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    url = f"http://localhost:{port}/"
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except Exception:
            time.sleep(0.3)
    return False


def main() -> None:
    tmp_dir = tempfile.mkdtemp(prefix="bartascope_streamlit_probe_")
    app_path = tmp_dir + "/scratch_streamlit_app.py"
    with open(app_path, "w") as f:
        f.write(TRIVIAL_APP)

    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", app_path,
         "--server.headless", "true",
         "--server.port", str(PORT),
         "--browser.gatherUsageStats", "false"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    try:
        up = wait_for_server(PORT)
        if not up:
            print("server never came up; stderr follows:")
            print(proc.stdout.read())
            return

        # Let it settle past initial request handling.
        time.sleep(6)
        exit_code = proc.poll()
        print(f"child pid={proc.pid} exit_code={exit_code} create_time_ok={psutil.pid_exists(proc.pid)}")
        p = psutil.Process(proc.pid)
        print(f"psutil sees: name={p.name()} status={p.status()} cmdline={p.cmdline()}")
        mem = p.memory_info()
        rss_mb = mem.rss / 1e6
        peak_mb = getattr(mem, "peak_wset", mem.rss) / 1e6
        print(f"streamlit server, idle after startup: rss={rss_mb:.1f} MB   peak_wset={peak_mb:.1f} MB")
        children = p.children(recursive=True)
        total_rss = mem.rss
        total_peak = peak_mb * 1e6
        for c in children:
            cmem = c.memory_info()
            c_peak = getattr(cmem, "peak_wset", cmem.rss)
            print(f"  child pid={c.pid} name={c.name()} rss={cmem.rss/1e6:.1f} MB peak_wset={c_peak/1e6:.1f} MB")
            total_rss += cmem.rss
            total_peak += c_peak
        print(f"TOTAL across process tree: rss={total_rss/1e6:.1f} MB   peak_wset={total_peak/1e6:.1f} MB")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
