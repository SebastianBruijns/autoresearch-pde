"""Sandboxed Python interpreter tool for the agent (exploratory analysis).

The agent's code runs in a fresh subprocess with a timeout. Pre-loaded names:
    meta, data        public dataset in the ACTIVE coordinates (dicts; data["U"], data["t"], ...)
    np, sp, plt       numpy, sympy, matplotlib.pyplot (Agg backend)
    tb, co            eqdisc.toolbox, eqdisc.coordinates
    WORK              pathlib.Path of the session workspace (save files / figures here)
Figures saved as PNG in WORK during the call are returned to the agent as images.
Two layers keep the hidden test set out of reach. (1) Where bubblewrap is installed (Linux), the code runs in a
sandbox (eqdisc.sandbox) with no network, in which only the eqdisc package, the inputs and the workspace exist.
(2) Always, after the prelude, an audit hook denies any file access inside the repository (datasets, hidden truth,
run outputs, other sessions) except the workspace and the eqdisc package source, and denies spawning processes; the
script runs from a temporary directory.
"""
import json
import os
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
# then (appended in run_code): the allow-list guard from eqdisc.guard -- the agent's code may read only its
# workspace, its temp dir, the Python installation and the eqdisc package minus files holding reference equations;
# hidden test data, ground truth and dataset caches are unreadable wherever they are on disk.


def run_code(code, meta, data, workdir, timeout=150, max_output=6000):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    before = {p: p.stat().st_mtime for p in workdir.glob("*")}
    meta = {k: v for k, v in meta.items() if k not in ("system", "name")}   # names can reveal the system
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "in.pkl").write_bytes(pickle.dumps((meta, data)))
        from .guard import allow_dirs, package_deny, prelude
        pkg = str(ROOT / "eqdisc")
        # inside bubblewrap the inputs are at /in, the workspace at /work and the venv at /opt/venv (eqdisc keeps its path)
        inside = [sandbox.WORK, "/in", sandbox.VENV, "/tmp"] if sandbox.available() else [tmp, workdir]
        guard = prelude(allow_dirs([*inside, pkg]), package_deny(pkg))
        (tmp / "script.py").write_text(PRELUDE % str(ROOT) + guard + "\n" + code)
        t0 = time.time()
        if sandbox.available():      # only the eqdisc package, the inputs and the workspace are visible
            argv, env = sandbox.wrap([sys.executable, "/in/script.py", "/in/in.pkl", sandbox.WORK], workdir,
                                     ro=[ROOT / "eqdisc"], binds={"/in": tmp})
        else:
            argv, env = sandbox.wrap([sys.executable, str(tmp / "script.py"), str(tmp / "in.pkl"), str(workdir)], workdir)
        try:
            p = subprocess.run(argv, cwd=workdir if sandbox.available() else tmp, env=env, capture_output=True,
                               text=True, timeout=timeout)
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
