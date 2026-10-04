"""Send a video to Claude on Modal: one bare API call with the video's frames and a prompt (default: no system prompt,
no tools, no other text). Frames are decoded here (eqdisc.video, in memory) and passed to the container.

    modal run send_video.py --args "--video clip.mov --prompt 'gimme PDE'"
    modal run send_video.py --args "--video clip.mov --prompt '...' --model claude-sonnet-5-5 --max-frames 60 --width 640"

Output: runs/video_<time>_<name>/bare.json (thinking, text, usage, cost, the full raw API response), bare.html (the same
as an attempt page) and frames_sent.json (which frames went out, at which timestamps).
"""
import argparse
import datetime as dt
import json
import shlex
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--prompt", default="gimme pde")
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--effort", default="high")
    p.add_argument("--max-frames", type=int, default=100, help="evenly spaced frames sent (API limit 100)")
    p.add_argument("--width", type=int, default=768, help="frame width in px (aspect kept; never upscaled)")
    p.add_argument("--start", type=float, default=None, help="clip start (s)")
    p.add_argument("--end", type=float, default=None, help="clip end (s)")
    p.add_argument("--timestamps", action="store_true", help="label each frame with its time (adds context)")
    p.add_argument("--name", default="", help="suffix for the output folder")
    return p.parse_args(argv)


def remote_ask(payload, cfg):
    from eqdisc.video import ask
    return ask(payload, cfg["prompt"], model=cfg["model"], effort=cfg["effort"], timestamps=cfg["timestamps"])


def drive(argv, ask_fn):
    from eqdisc.video import load_video
    a = parse(argv)
    video = Path(a.video).expanduser().resolve()
    v = load_video(video, max_frames=a.max_frames, width=a.width, start=a.start, end=a.end)
    tag = a.name or video.stem
    out = ROOT / "runs" / f"video_{time.strftime('%Y%m%d-%H%M%S')}_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"prompt": a.prompt, "model": a.model, "effort": a.effort, "timestamps": a.timestamps}
    (out / "config.json").write_text(json.dumps({"video": str(video), "argv": argv, **cfg,
                                                 "started": dt.datetime.now().isoformat()}, indent=1))
    (out / "frames_sent.json").write_text(json.dumps({**v.summary(), "times": v.times}, indent=1))
    print(f"{video.name}: sending {len(v.jpegs)} frames {v.size[0]}x{v.size[1]} "
          f"({v.summary()['bytes'] / 1e6:.1f} MB) + prompt {a.prompt!r} to {a.model}; output {out}", flush=True)
    try:
        res = ask_fn(v, cfg)
    except Exception as e:  # noqa: BLE001
        res = {"error": f"{type(e).__name__}: {e}", "log": []}
    res = {"problem": "video", "agent": "bare", "skills": "off", **res,
           "submitted": {"expr": res.get("answer", "")[:4000], "rationale": ""},
           "truth": "(none: no ground truth for this video)"}
    (out / "bare.json").write_text(json.dumps(res, indent=1, default=str))
    try:
        from types import SimpleNamespace as NS
        import run_bench as RB
        run = NS(a=NS(target=video.name, model=a.model, effort=a.effort, kind="video", skills_label="off"))
        (out / "bare.html").write_text(RB.attempt_page(run, "bare", res).replace("../dashboard.html", "bare.json"))
    except Exception as e:  # noqa: BLE001
        print(f"(html page not written: {e})")
    print(f"done: ${res.get('cost_usd', 0)}, stop={res.get('stop')}, {res.get('wall_s')} s"
          f"{' ERROR ' + res['error'] if res.get('error') else ''}\nlogs: {out}", flush=True)


try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    import os as _os
    _secret = _os.environ.get("EQDISC_MODAL_SECRET", "anthropic-api-key")
    image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("numpy>=1.26", "scipy>=1.11", "sympy>=1.12", "anthropic>=1.11")
             .add_local_python_source("eqdisc", ignore=["**/*.pyc", "**/tests/**"]))
    app = modal.App("eqdisc-send-video", image=image)

    @app.function(secrets=[modal.Secret.from_name(_secret)], timeout=3600, cpu=1.0, memory=2048)
    def ask_remote(payload, cfg):
        return remote_ask(payload, cfg)

    @app.local_entrypoint()
    def main(args: str = ""):
        drive(shlex.split(args), lambda v, c: ask_remote.remote(v, c))
