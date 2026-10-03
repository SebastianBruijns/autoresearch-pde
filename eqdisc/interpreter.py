"""Sandboxed Python interpreter tool for the agent (KeplerAgent-style exploratory analysis).

The agent's code runs in a fresh subprocess with a timeout. Pre-loaded names:
    meta, data        public dataset in the ACTIVE coordinates (dicts; data["U"], data["t"], ...)
    np, sp, plt       numpy, sympy, matplotlib.pyplot (Agg backend)
    tb, co            eqdisc.toolbox, eqdisc.coordinates
    WORK              pathlib.Path of the session workspace (save files / figures here)
Figures saved as PNG in WORK during the call are returned to the agent as images.
The code runs in a bubblewrap sandbox (eqdisc.sandbox): no network, and only the eqdisc package, the inputs and
the workspace are visible, so the hidden test set is not reachable.
"""
import json
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import sandbox

ROOT = Path(__file__).resolve().parent.parent

PRELUDE = """
import json, pickle, sys, math
sys.path.insert(0, %r)
from pathlib import Path
import numpy as np, sympy as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from eqdisc import toolbox as tb, coordinates as co
meta, data = pickle.loads(Path(sys.argv[1]).read_bytes())
WORK = Path(sys.argv[2])
"""


def run_code(code, meta, data, workdir, timeout=600, max_output=6000):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    before = {p: p.stat().st_mtime for p in workdir.glob("*")}
    meta = {k: v for k, v in meta.items() if k not in ("system", "name")}   # names can reveal the system
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "in.pkl").write_bytes(pickle.dumps((meta, data)))
        (tmp / "script.py").write_text(PRELUDE % str(ROOT) + "\n" + code)
        t0 = time.time()
        if sandbox.available():      # only the eqdisc package, the inputs and the workspace are visible
            argv, env = sandbox.wrap([sys.executable, "/in/script.py", "/in/in.pkl", sandbox.WORK], workdir,
                                     ro=[ROOT / "eqdisc"], binds={"/in": tmp})
        else:
            argv, env = sandbox.wrap([sys.executable, str(tmp / "script.py"), str(tmp / "in.pkl"), str(workdir)], workdir)
        try:
            p = subprocess.run(argv, cwd=workdir, env=env, capture_output=True, text=True, timeout=timeout)
            out, err, rc = p.stdout, p.stderr, p.returncode
        except subprocess.TimeoutExpired as e:
            out, err, rc = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), \
                f"TIMEOUT after {timeout}s", -1
    new = [p for p in workdir.glob("*") if p.stat().st_mtime > before.get(p, 0)]
    trim = lambda s: s if len(s) <= max_output else s[: max_output // 2] + "\n...[truncated]...\n" + s[-max_output // 2:]
    return {"returncode": rc, "stdout": trim(out), "stderr": trim(err[-3000:]) if rc else "",
            "seconds": round(time.time() - t0, 2),
            "new_files": [str(p.name) for p in new],
            "images": [str(p) for p in new if p.suffix.lower() == ".png"]}
