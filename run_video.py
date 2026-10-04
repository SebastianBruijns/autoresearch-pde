"""Two parallel Modal runs on a video, with full logs (tool calls + reasoning summaries):

  bare    : one bare Claude API call. Content = the video's frames (in order, as images) + the text "gimme pde".
            No system prompt, no tools, no other text (the API takes images, not video files).
  harness : the eqdisc agent (playbook + tools + critic) on the video turned into an eqdisc dataset (RGB intensity
            fields r, g, b on a downsampled pixel grid over time; the clip is split into 3 consecutive segments so the
            harness can hold one out), instructed: "gimme pde" + use the harness tools.

    modal run run_video.py --args "--video clip.mov"
Frames are decoded on this machine (ffmpeg) in memory and sent to the containers; nothing is written except the logs in
runs/video_<time>/.
"""
import argparse
import datetime as dt
import json
import shlex
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--effort", default="high")
    p.add_argument("--bare-frames", type=int, default=60, help="frames sent to the bare call (evenly spaced)")
    p.add_argument("--bare-width", type=int, default=768)
    p.add_argument("--grid-width", type=int, default=96, help="harness dataset grid width (height keeps aspect)")
    p.add_argument("--max-tools", type=int, default=30)
    p.add_argument("--max-cost", type=float, default=5.0)
    p.add_argument("--max-minutes", type=float, default=45.0)
    return p.parse_args(argv)


# ----------------------------------------------------------------------------- the two runs (inside containers)
def bare_call(jpegs, cfg):
    """One bare API call: frames + 'gimme pde', nothing else (eqdisc.video.ask, the same call as send_video.py)."""
    from eqdisc.video import VideoPayload, ask
    v = VideoPayload(jpegs=list(jpegs), times=[], fps=cfg.get("fps", 0.0), size=(0, 0))
    r = ask(v, "gimme pde", model=cfg["model"], effort=cfg["effort"])
    return {"problem": "video", "agent": "bare", "skills": "off", **r, "n_frames_sent": len(jpegs),
            "submitted": {"expr": r["answer"][:4000], "rationale": ""},
            "truth": "(none: no ground truth for this video)"}


def harness_run(frames, fps, cfg):
    """The eqdisc harness agent on the video as an eqdisc PDE dataset."""
    import numpy as np
    from eqdisc.agent import make_client, run_agent
    from eqdisc.solvers import derivative_symbols
    from eqdisc.srsd import compact_log
    t0 = time.time()
    U = frames.astype(np.float64) / 255.0                 # (nt, h, w, 3)
    U = np.transpose(U, (0, 2, 1, 3))                    # (nt, x=w, y=h, 3): x left->right, y top->bottom
    nseg = 3
    L = U.shape[0] // nseg
    U = np.stack([U[k * L:(k + 1) * L] for k in range(nseg)])   # 3 consecutive segments as trajectories
    nx, ny = U.shape[2], U.shape[3]
    import tempfile
    base = Path(tempfile.mkdtemp(prefix="w"))          # neutral random path: nothing visible names the data's origin
    ds = base / "d"
    ds.mkdir(parents=True, exist_ok=True)
    t = np.arange(L) / fps
    x, y = np.arange(nx, dtype=float), np.arange(ny, dtype=float)
    np.savez(ds / "data.npz", t=t, U=U, x=x, y=y)
    variables, dims = ["r", "g", "b"], ["x", "y"]
    meta = {"name": "d", "kind": "pde", "variables": variables, "dt": float(1 / fps), "n_traj": nseg,
            "shape": list(U.shape), "shape_doc": "(n_traj, nt, nx, ny, n_fields)", "system": None,
            "L": float(nx - 1), "nx": nx, "boundary": "neumann", "spatial_dims": dims,
            "grid": {"x": {"n": nx, "L": float(nx - 1), "x0": 0.0}, "y": {"n": ny, "L": float(ny - 1), "x0": 0.0}},
            "allowed_symbols": derivative_symbols(variables, dims, 4), "noise": None}
    (ds / "meta.json").write_text(json.dumps(meta))
    r = run_agent(ds, model=cfg["model"], effort=cfg["effort"], max_tools=cfg["max_tools"], out_dir=base / "r",
                  verbose=True, client=make_client(), critic=True, use_memory=False, report=False, judge_llm=False,
                  context="gimme pde. Use the tools of this harness to work it out.", final_assessment=False,
                  max_cost_usd=cfg["max_cost"], max_wall_s=cfg["max_minutes"] * 60)
    try:
        log = compact_log(json.loads((Path(r["out_dir"]) / "transcript.json").read_text()))
    except Exception:  # noqa: BLE001
        log = []
    sub = r.get("submitted")
    return {"problem": "video", "agent": "harness", "skills": "off", "stop": r.get("stop"),
            "n_tool_calls": r["n_tool_calls"], "usage": r["usage"], "cost_usd": r["cost_usd"], "log": log,
            "critic": r.get("critic", []), "dataset": {"shape": list(U.shape), "fps": fps},
            "submitted": {"expr": json.dumps(sub["rhs"]), "rationale": sub.get("rationale", "")} if sub else None,
            "wall_s": round(time.time() - t0, 1), "truth": "(none: no ground truth for this video)"}


# ----------------------------------------------------------------------------- local driver
def save(out, name, res, a):
    from types import SimpleNamespace as NS
    import run_bench as RB
    (out / f"{name}.json").write_text(json.dumps(res, indent=1, default=str))
    run = NS(a=NS(target=Path(a.video).name, model=a.model, effort=a.effort, kind="video", skills_label="off"))
    (out / f"{name}.html").write_text(RB.attempt_page(run, name, res).replace("../dashboard.html", "index.html"))


def drive(argv, bare_fn, harness_fn):
    a = parse(argv)
    video = Path(a.video).expanduser().resolve()
    out = ROOT / "runs" / f"video_{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    from eqdisc.video import decode_jpegs, decode_rgb     # frame decoding shared with send_video.py
    jpegs_all, fps, _ = decode_jpegs(video, a.bare_width)
    step = max(1, len(jpegs_all) // a.bare_frames)
    jpegs = jpegs_all[::step][:a.bare_frames]
    frames, _ = decode_rgb(video, a.grid_width)
    cfg = {"model": a.model, "effort": a.effort, "max_tools": a.max_tools, "max_cost": a.max_cost,
           "max_minutes": a.max_minutes, "fps": fps}
    print(f"video {video.name}: {len(jpegs_all)} frames at {fps:.3g} fps; bare gets {len(jpegs)} frames "
          f"({a.bare_width}px); harness grid {frames.shape[2]}x{frames.shape[1]}; output {out}", flush=True)
    (out / "config.json").write_text(json.dumps({"video": str(video), "argv": argv, **cfg, "bare_frames": len(jpegs),
                                                 "frame_step": step, "fps": fps, "started": dt.datetime.now().isoformat()}))
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(2) as ex:
        futs = {"bare": ex.submit(bare_fn, jpegs, cfg), "harness": ex.submit(harness_fn, frames, fps, cfg)}
        for name, f in futs.items():
            try:
                res = f.result()
            except Exception as e:  # noqa: BLE001
                res = {"problem": "video", "agent": name, "error": f"{type(e).__name__}: {e}", "log": []}
            save(out, name, res, a)
            print(f"{name}: done, ${res.get('cost_usd', 0)}, {res.get('n_tool_calls', 0)} tool calls, "
                  f"stop={res.get('stop')}{' ERROR ' + res['error'] if res.get('error') else ''}", flush=True)
    links = "".join(f"<li><a href='{n}.html'>{n}</a></li>" for n in ("bare", "harness"))
    (out / "index.html").write_text(f"<!doctype html><meta charset=utf-8><title>video runs</title>"
                                    f"<h1>{video.name}</h1><ul>{links}</ul>")
    print(f"logs: {out}/index.html")


try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    import os as _os
    _secret = _os.environ.get("EQDISC_MODAL_SECRET", "anthropic-api-key")
    # same image steps as run_bench.py (identical layers are reused from Modal's cache)
    image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("numpy>=1.26", "scipy>=1.11", "sympy>=1.12", "matplotlib>=3.8", "pandas>=2.0",
                          "pysindy>=2.0", "anthropic>=1.11", "pysr>=1.0", "h5py", "huggingface_hub", "fsspec")
             .run_commands('python -c "import numpy as np; from pysr import PySRRegressor; '
                           'PySRRegressor(niterations=1, progress=False, verbosity=0).fit(np.random.rand(30, 2), np.random.rand(30))"')
             .add_local_python_source("eqdisc", ignore=["**/*.pyc", "**/tests/**"]))
    app = modal.App("eqdisc-video", image=image)

    @app.function(secrets=[modal.Secret.from_name(_secret)], timeout=3600, cpu=2.0, memory=8192)
    def bare_remote(jpegs, cfg):
        return bare_call(jpegs, cfg)

    @app.function(secrets=[modal.Secret.from_name(_secret)], timeout=3 * 3600, cpu=4.0, memory=16384)
    def harness_remote(frames, fps, cfg):
        return harness_run(frames, fps, cfg)

    @app.local_entrypoint()
    def main(args: str = ""):
        drive(shlex.split(args), lambda j, c: bare_remote.remote(j, c), lambda f, fps, c: harness_remote.remote(f, fps, c))
