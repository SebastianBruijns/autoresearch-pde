"""Run a tool function in a separate process with a hard wall-clock limit.

Anything that can hang or crash the interpreter (PySR/Julia, sympy simplification of huge
expressions, agent-written code) goes through here, so one bad call can't stall a session.

    out = run_isolated("eqdisc.toolbox", "run_pysr", meta, data, {"target": "omega"}, timeout=300)
"""
import importlib
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_isolated(module, fn, meta, data, kwargs, timeout=300):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "in.pkl").write_bytes(pickle.dumps((module, fn, meta, data, kwargs)))
        try:
            p = subprocess.run([sys.executable, "-m", "eqdisc.isolate", str(tmp)], cwd=ROOT,
                               capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"error": f"{fn} timed out after {timeout}s (process killed); try a smaller budget/search space"}
        out = tmp / "out.pkl"
        if not out.exists():
            return {"error": f"{fn} crashed: {p.stderr.strip()[-1500:]}"}
        return pickle.loads(out.read_bytes())


def _worker(tmp):
    tmp = Path(tmp)
    module, fn, meta, data, kwargs = pickle.loads((tmp / "in.pkl").read_bytes())
    try:
        res = getattr(importlib.import_module(module), fn)(meta, data, **kwargs)
    except Exception as e:  # noqa: BLE001
        import traceback
        res = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1200:]}
    (tmp / "out.pkl").write_bytes(pickle.dumps(res))


if __name__ == "__main__":
    _worker(sys.argv[1])
