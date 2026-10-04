"""Run agent-written code in a bubblewrap sandbox, so it cannot read hidden test data or reach the network.

Inside the sandbox only the OS (/usr, /etc), the Python installation, explicitly listed read-only paths and the
writable work directory (mounted at /work) exist. The repository, datasets/*/hidden and the home directory do not.
Without bwrap (e.g. macOS) only the interpreter's audit-hook guard applies, with a warning (set
EQDISC_REQUIRE_SANDBOX=1 to make that an error).
"""
import os
import shutil
import sys
import warnings
from pathlib import Path

WORK = "/work"
VENV = "/opt/venv"


def _python_roots():
    """Directories the interpreter needs: the venv and the base installation it was created from."""
    roots = {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve(), Path(sys.executable).resolve().parent.parent}
    return sorted(str(r) for r in roots if not str(r).startswith("/usr"))


def available():
    return shutil.which("bwrap") is not None


def wrap(cmd, workdir, ro=(), binds=None):
    """Return (argv, env) running `cmd` sandboxed, with cwd /work = workdir. ro: extra read-only paths (same
    location inside). binds: {inside_path: outside_path} extra read-only mounts. Paths in cmd are inside paths when
    available(), real paths otherwise."""
    env = {"PATH": "/usr/bin:/bin", "HOME": WORK, "MPLBACKEND": "Agg", "MPLCONFIGDIR": "/tmp/mpl",
           "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "4")}
    if not available():
        if os.environ.get("EQDISC_REQUIRE_SANDBOX") == "1":
            raise RuntimeError("bwrap not found and EQDISC_REQUIRE_SANDBOX=1")
        warnings.warn("bwrap not found: agent code runs without the bubblewrap sandbox; the interpreter's audit-hook "
                      "guard still blocks repository file reads and subprocesses")
        return list(cmd), {**env, "HOME": str(workdir)}      # callers pass real paths when bwrap is missing
    argv = ["bwrap", "--ro-bind", "/usr", "/usr", "--ro-bind", "/etc", "/etc",
            "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
            "--unshare-all", "--die-with-parent", "--new-session"]
    venv = Path(sys.prefix).resolve()
    if sys.prefix != sys.base_prefix:     # mount the venv at a neutral path: its location names the project
        argv += ["--ro-bind", str(venv), VENV]
        cmd = [VENV + "/bin/python" if c == sys.executable else c for c in cmd]
    for p in [*(r for r in _python_roots() if Path(r) != venv), *map(str, ro)]:
        argv += ["--ro-bind", p, p]
    for inside, outside in (binds or {}).items():
        argv += ["--ro-bind", str(outside), str(inside)]
    argv += ["--bind", str(Path(workdir).resolve()), WORK, "--chdir", WORK, "--"]
    return argv + list(cmd), env
