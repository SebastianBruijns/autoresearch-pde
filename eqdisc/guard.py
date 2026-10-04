"""Allow-list guard for code written by an agent, installed as a sys.audit hook in its subprocess.

It complements eqdisc.sandbox (bubblewrap): bwrap isolates the process where it is installed (Linux); this guard is
always on, and is the only protection where bwrap is missing (macOS, Modal containers).

Reads/listings are allowed only inside the allowed directories (the agent's own folders, the Python installation,
optionally the eqdisc package minus files that hold reference equations) and a few system paths; process spawning,
links and native calls through ctypes (which bypass Python's file audit events) are denied. Everything else (hidden test data, ground-truth files, dataset caches, other runs, the repository) is denied,
wherever it lives on disk.
"""
import os
import sys
import sysconfig

# eqdisc source files that contain reference equations or the scoring machinery
SECRET_FILES = ("hf_well.py", "mhd_sim.py", "judge.py", "systems.py", "blind.py", "evaluate.py", "benchmark.py",
                "sr_bench.py", "sr_variants.py", "datagen.py", "srsd.py", "bare.py", "well_gs.py")

GUARD = r"""
def _sandbox(ALLOW, DENY):
    import os, sys
    from pathlib import Path
    try:                                  # load ctypes now (its import opens libpython), then deny every native call:
        import ctypes                     # a ctypes call to libc fopen/fread makes no Python "open" audit event
    except Exception:
        pass
    ALLOW = [Path(a).resolve() for a in ALLOW]
    DENY = [Path(d).resolve() for d in DENY]
    def inside(p, d):
        try:
            Path(p).resolve().relative_to(d)
            return True
        except Exception:
            return False
    def hook(event, args):
        if event.startswith("ctypes."):      # dlopen, dlsym, call_function, cdata, string_at, ...
            raise PermissionError("native calls (ctypes) are disabled in this sandbox")
        if event in ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty",
                     "pty.spawn", "os.startfile"):
            raise PermissionError("process spawning is disabled in this sandbox")
        if event in ("os.link", "os.symlink", "os.rename", "os.replace", "os.truncate", "shutil.copyfile",
                     "shutil.copytree", "shutil.move") and args:
            paths = [a for a in args[:2] if isinstance(a, (str, bytes, os.PathLike))]
            for q in paths:                   # both ends of a link/rename/copy must stay inside the sandbox
                q = os.fsdecode(q)
                if any(inside(q, d) for d in DENY) or not any(inside(q, a) for a in ALLOW):
                    raise PermissionError(f"access outside the sandbox is denied: {q}")
            if event in ("os.link", "os.symlink"):
                raise PermissionError("creating links is disabled in this sandbox")
            return
        if event in ("open", "os.listdir", "os.scandir", "glob.glob", "os.chdir", "os.walk") and args:
            p = args[0]
            if isinstance(p, int) or p is None:
                return
            p = os.fsdecode(p) if isinstance(p, (str, bytes, os.PathLike)) else str(p)
            if any(inside(p, d) for d in DENY) or not any(inside(p, a) for a in ALLOW):
                raise PermissionError(f"access outside the sandbox is denied: {p}")
    sys.addaudithook(hook)
"""


def allow_dirs(extra=()):
    """Python installation + system library paths needed for imports, plus `extra` (the agent's own folders)."""
    dirs = {sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix}
    for k in ("stdlib", "platstdlib", "purelib", "platlib", "include", "scripts", "data"):
        try:
            dirs.add(sysconfig.get_paths()[k])
        except KeyError:
            pass
    for p in sys.path:
        if p and ("site-packages" in p or "dist-packages" in p or p.startswith(sys.prefix)):
            dirs.add(p)
    dirs |= {"/usr/lib", "/usr/lib64", "/usr/share", "/usr/local/lib", "/etc/localtime", "/etc/ssl",
             "/dev/null", "/dev/urandom", "/proc/self", "/System/Library", "/Library/Frameworks",
             # system fonts (matplotlib scans them): not secrets
             "/usr/X11R6", "/opt/X11", "/Library/Fonts", os.path.expanduser("~/Library/Fonts"), "/usr/share/fonts"}
    try:                                   # matplotlib's config/cache dirs and every system font dir it scans
        import matplotlib
        from matplotlib import font_manager as fm
        dirs.add(matplotlib.get_cachedir())
        dirs.add(matplotlib.get_configdir())
        for name in ("X11FontDirectories", "OSXFontDirectories", "MSFontDirectories", "MSUserFontDirectories"):
            dirs |= {os.path.expanduser(d) for d in getattr(fm, name, [])}
    except Exception:  # noqa: BLE001
        pass
    return [d for d in dirs | set(map(str, extra)) if d]


def prelude(allow, deny=()):
    """Python source that installs the sandbox; prepend it to the agent's script (after any trusted setup code)."""
    return GUARD + f"\n_sandbox({list(map(str, allow))!r}, {list(map(str, deny))!r})\ndel _sandbox\n"


def package_deny(pkg_dir):
    """Paths inside the eqdisc package that agent code must not read."""
    return [os.path.join(pkg_dir, f) for f in SECRET_FILES]
