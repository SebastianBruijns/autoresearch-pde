"""Track caller-supplied points through a video with DUSTrack's Lucas-Kanade tracker and write them to a CSV.

    python tracker-dusttrack/track_video.py VIDEO POINTS [-o tracks.csv] [--overlay tracks.png] [--annotated tracks.mp4]

Run it with the interpreter of the DUSTrack environment (tracker-dusttrack/.venv, see README section below).

Pipeline:
  1. Read the clip (optionally a segment) with OpenCV.
  2. Remove camera shake: register every frame to a static-background image (same method as
     tracker/track_video.py, whose helper functions are imported). Skip with --no-stabilise for tripod footage.
  3. Track every point of POINTS through the stabilised frames with DUSTrack (dustrack 1.3.x):
       - one start position: dustrack.lk_filter.lucas_kanade_2 forward to the end of the clip and backward
         to its start (frame-to-frame pyramidal Lucas-Kanade);
       - several positions of the same point at different frames ("keyframes"): between two keyframes
         dustrack.lk_filter.lucas_kanade_rstc_2, which blends a forward and a backward pass so the track
         passes through both keyframes; outside the first/last keyframe as above.
  4. Optional quality gate (--fb-threshold, not part of DUSTrack): DUSTrack never reports a lost point, it
     returns a position for every frame. The gate re-tracks each step backwards and blanks a point from
     the first step whose forward-backward error exceeds the threshold, up to the next keyframe.

POINTS file: a CSV with a header line. Columns:
    x, y     required. Position in pixels of the ORIGINAL video frame (x right, y down), as read off a frame
             saved from the video (e.g. with --dump-frame).
    frame    optional. Frame number (0-based, counted from the start of the video file) at which x, y was
             read. Default: the first processed frame.
    point    optional. Integer 1, 2, ... (or p1, p2, ...) naming the point. Default: row number, so every row
             is a new point. Give the same number on several rows to add keyframes for one point.
Blank lines and lines starting with # are ignored. Output columns are p1, p2, ... in order of point number.

The output CSV has one row per frame: frame, time_s, then p1_x, p1_y, p2_x, p2_y, ... in pixels of the original
video (x right, y down), in the camera-stabilised coordinates of the background image (identical to raw frame
coordinates with --no-stabilise). Cells are empty where a point was not tracked in that frame.

Environment (DUSTrack needs numpy<2 and Python <= 3.13, so it has its own virtualenv):
    uv venv tracker-dusttrack/.venv --python 3.12
    uv pip install --python tracker-dusttrack/.venv/bin/python DUSTrack trackpy
(trackpy is only needed because tracker/track_video.py imports it at module level.)
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tracker"))
import track_video as base  # noqa: E402  (tracker/track_video.py: read_frames, stabilisation helpers, drawing)


def load_dustrack():
    """Import DUSTrack's frame-list Lucas-Kanade functions. The package imports its Qt GUI modules on import,
    so force an off-screen Qt platform; its start-up chatter on stdout is sent to stderr."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    with warnings.catch_warnings(), contextlib.redirect_stdout(sys.stderr):
        warnings.simplefilter("ignore")
        from dustrack.lk_filter import lucas_kanade_2, lucas_kanade_rstc_2
    return lucas_kanade_2, lucas_kanade_rstc_2


def read_points(path, default_frame):
    """-> {point number: [(frame, x, y), ...] sorted by frame}"""
    df = pd.read_csv(path, comment="#", skip_blank_lines=True, skipinitialspace=True)
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"x", "y"} <= set(df.columns):
        sys.exit(f"{path}: need a header line with at least the columns x,y (optional: frame, point)")
    if "frame" not in df:
        df["frame"] = default_frame
    df["frame"] = df["frame"].fillna(default_frame).astype(int)
    if "point" not in df:
        df["point"] = np.arange(1, len(df) + 1)
    df["point"] = df["point"].astype(str).str.strip().str.lstrip("pP").astype(int)
    points = {}
    for pid, g in df.sort_values(["point", "frame"]).groupby("point"):
        if g["frame"].duplicated().any():
            sys.exit(f"{path}: point {pid} has two positions for the same frame")
        points[int(pid)] = [(int(r.frame), float(r.x), float(r.y)) for r in g.itertuples()]
    if not points:
        sys.exit(f"{path}: no points")
    return points


def stabilise(frames, log):
    """Same two-pass registration as tracker/track_video.py stabilise(), built from its helper functions,
    but also returning the per-frame transforms (stabilised coords -> frame coords) so that start points
    given in raw frame coordinates can be mapped into stabilised coordinates."""
    gray = base.to_gray(frames)
    static = ((1 - base.moving_region(gray)) * 255).astype(np.uint8)
    transforms, support = base.register(gray[0], gray, static, allow_rotation=False)
    if transforms is None:
        log("  no static background features found; frames left unstabilised")
        return frames, [np.eye(2, 3)] * len(frames)
    background = cv2.cvtColor(base.median_image(base.warp(frames, transforms)), cv2.COLOR_BGR2GRAY)
    transforms2, support2 = base.register(background, gray, None, allow_rotation=True)
    if transforms2 is not None:
        transforms, support = transforms2, support2
    shifts = np.array([M[:, 2] for M in transforms])
    log(f"  camera shift range x {np.ptp(shifts[:, 0]):.1f} px, y {np.ptp(shifts[:, 1]):.1f} px (processed scale); "
        f"background features per frame: median {int(np.median(support))}, min {support.min()}")
    return base.warp(frames, transforms), transforms


def to_stabilised(xy, M):
    """Frame coordinates -> stabilised coordinates (M maps stabilised -> frame)."""
    return np.linalg.solve(M[:, :2], np.asarray(xy, dtype=np.float64) - M[:, 2])


def track_point(gray, keys, lk2, rstc2, lk_config):
    """keys: [(local frame index, xy)], sorted. -> (n_frames, 2) array, a position for every frame."""
    n = len(gray)
    out = np.full((n, 2), np.nan)
    k0, xy0 = keys[0]
    if k0 > 0:  # backward from the first keyframe to the start of the clip
        out[k0::-1] = lk2(gray[k0::-1], xy0[None], **lk_config)[:, 0]
    for (ka, xya), (kb, xyb) in zip(keys[:-1], keys[1:]):  # keyframe to keyframe, forward/backward blend
        out[ka:kb + 1] = rstc2(gray[ka:kb + 1], xya[None], xyb[None], **lk_config)[:, 0]
    k1, xy1 = keys[-1]
    out[k1:] = lk2(gray[k1:], xy1[None], **lk_config)[:, 0]
    return out


def fb_gate(gray, track, keys, threshold, lk_config):
    """Blank the parts of a track that fail a forward-backward check. Each step a -> b of the track is
    re-tracked b -> a; if that misses the position at a by more than `threshold` px, or the point left the
    image, the track is cut there. Cuts run away from keyframes: a segment between two keyframes is
    blanked only between its first failing step from either side."""
    n = len(gray)
    h, w = gray[0].shape
    err = np.zeros(n - 1)  # err[i]: step between frames i and i+1
    cfg = dict(winSize=(45, 45), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    cfg.update(lk_config)
    for i in range(n - 1):
        a, b = track[i], track[i + 1]
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        if not (0 <= b[0] < w and 0 <= b[1] < h and 0 <= a[0] < w and 0 <= a[1] < h):
            err[i] = np.inf
            continue
        back, ok, _ = cv2.calcOpticalFlowPyrLK(gray[i + 1], gray[i], b.astype(np.float32).reshape(1, 1, 2), None, **cfg)
        fwd, ok2, _ = cv2.calcOpticalFlowPyrLK(gray[i], gray[i + 1], a.astype(np.float32).reshape(1, 1, 2), None, **cfg)
        e = max(np.linalg.norm(back[0, 0] - a), np.linalg.norm(fwd[0, 0] - b)) if ok[0, 0] and ok2[0, 0] else np.inf
        err[i] = e
    bad = np.flatnonzero(err > threshold)
    out = track.copy()
    kf = [k for k, _ in keys]
    first_bad_before = bad[bad < kf[0]]
    if len(first_bad_before):  # backward part: blank everything before the last bad step
        out[:first_bad_before.max() + 1] = np.nan
    for ka, kb in zip(kf[:-1], kf[1:]):
        seg = bad[(bad >= ka) & (bad < kb)]
        if len(seg):
            out[seg.min() + 1:seg.max() + 1] = np.nan
    after = bad[bad >= kf[-1]]
    if len(after):
        out[after.min() + 1:] = np.nan
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video", type=Path)
    ap.add_argument("points", type=Path, nargs="?", help="CSV of start points: x,y[,frame][,point] (see module docstring)")
    ap.add_argument("-o", "--output", type=Path, help="CSV path (default: VIDEO name with _tracks.csv, in the current directory)")
    ap.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    ap.add_argument("--duration", type=float, default=None, help="seconds to process (default: to the end)")
    ap.add_argument("--scale", type=float, default=1.0, help="downscale factor for processing (default 1.0)")
    ap.add_argument("--win-size", type=int, default=45, help="Lucas-Kanade window in processed pixels (DUSTrack default 45)")
    ap.add_argument("--max-level", type=int, default=2, help="Lucas-Kanade pyramid levels above the base (DUSTrack default 2)")
    ap.add_argument("--fb-threshold", type=float, default=None,
                    help="blank a point from the first step whose forward-backward error exceeds this many original "
                         "pixels (default: off, DUSTrack output is written as is)")
    ap.add_argument("--no-stabilise", action="store_true", help="skip camera-shake removal (tripod footage)")
    ap.add_argument("--overlay", type=Path, help="write a PNG of the trajectories over the background")
    ap.add_argument("--annotated", type=Path, help="write an MP4 of the stabilised clip with the points marked")
    ap.add_argument("--dump-frame", type=int, nargs="+", metavar="N",
                    help="write original frame(s) N as PNG next to the output (for reading off start points) and exit")
    args = ap.parse_args(argv)
    output = args.output or Path(args.video.stem + "_tracks.csv")
    t0 = time.time()

    def log(msg):
        print(msg, file=sys.stderr)

    frames, fps, first = base.read_frames(args.video, args.start, args.duration, args.scale)
    n = len(frames)
    log(f"{n} frames at {fps:.2f} fps, processed at {frames[0].shape[1]}x{frames[0].shape[0]}")
    if args.dump_frame:
        for f in args.dump_frame:
            if not first <= f < first + n:
                sys.exit(f"frame {f} is outside the processed range {first}..{first + n - 1}")
            path = output.with_name(f"{args.video.stem}_frame{f:05d}.png")
            cv2.imwrite(str(path), frames[f - first])
            log(f"wrote {path}" + ("" if args.scale == 1.0 else f" (scaled by {args.scale}: divide coordinates by it)"))
        return 0
    if args.points is None:
        ap.error("POINTS file is required")
    points = read_points(args.points, first)
    for pid, keys in points.items():
        for f, _, _ in keys:
            if not first <= f < first + n:
                sys.exit(f"point {pid}: frame {f} is outside the processed range {first}..{first + n - 1}")

    if args.no_stabilise:
        transforms = [np.eye(2, 3)] * n
    else:
        frames, transforms = stabilise(frames, log)
    gray = base.to_gray(frames)

    lk2, rstc2 = load_dustrack()
    lk_config = dict(winSize=(args.win_size, args.win_size), maxLevel=args.max_level)
    x, y = {}, {}
    for pid, keys in points.items():
        local = [(f - first, to_stabilised(np.array([px, py]) * args.scale, transforms[f - first])) for f, px, py in keys]
        track = track_point(gray, local, lk2, rstc2, lk_config)
        raw = track
        if args.fb_threshold is not None:
            track = fb_gate(gray, track, local, args.fb_threshold * args.scale, lk_config)
        step = np.linalg.norm(np.diff(raw, axis=0), axis=1) / args.scale
        valid = int(np.isfinite(track[:, 0]).sum())
        log(f"  p{pid}: {len(keys)} keyframe(s) at frame(s) {[k[0] for k in keys]}; tracked {valid}/{n} frames; "
            f"step per frame median {np.median(step):.1f} px, max {step.max():.1f} px (at frame {first + int(step.argmax()) + 1})")
        x[pid], y[pid] = track[:, 0], track[:, 1]
    x, y = pd.DataFrame(x), pd.DataFrame(y)

    table = pd.DataFrame({"frame": first + np.arange(n), "time_s": (first + np.arange(n)) / fps})
    for pid in x.columns:
        table[f"p{pid}_x"] = x[pid] / args.scale
        table[f"p{pid}_y"] = y[pid] / args.scale
    table.to_csv(output, index=False, float_format="%.2f")
    if args.overlay or args.annotated:
        if list(x.columns) != list(range(1, x.shape[1] + 1)):
            log("  note: the drawing labels points p1, p2, ... in column order, not by their numbers in POINTS")
    if args.overlay:
        base.write_overlay(args.overlay, base.median_image(frames), x, y)
    if args.annotated:
        base.write_annotated(args.annotated, frames, fps, x, y)
    log(f"wrote {output} ({x.shape[1]} points, {n} frames) in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
