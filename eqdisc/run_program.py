"""Sandbox entry point: run an evolved program on a *public-only* dataset copy.

    python -m eqdisc.run_program program.py public_dir out.json
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np


def main():
    prog, d, out = sys.argv[1:4]
    spec = importlib.util.spec_from_file_location("candidate_program", prog)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    meta = json.loads((Path(d) / "meta.json").read_text())
    data = dict(np.load(Path(d) / "data.npz"))
    t0 = time.time()
    res = mod.discover(meta, data)
    res = res if "rhs" in res else {"rhs": res}
    res["rhs"] = {k: str(v) for k, v in res["rhs"].items()}
    res["runtime_s"] = time.time() - t0
    Path(out).write_text(json.dumps(res))


if __name__ == "__main__":
    main()
