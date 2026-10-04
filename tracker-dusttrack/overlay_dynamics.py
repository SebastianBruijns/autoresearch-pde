"""Draw tracked points and the motion predicted by a discovered model onto the original video.

    python tracker-dusttrack/overlay_dynamics.py VIDEO TRACKS_CSV RESULT_JSON [-o OUT] [--data DATASET]
        [--sim-csv CSV] [--tail 25] [--link p3:p1 p4:p2] [--report-times 1 5 10]

TRACKS_CSV has the columns frame, time_s, p1_x, p1_y, ... in original-video pixels. RESULT_JSON is a run's
result.json; its submitted -> rhs maps every state variable (pK_x, pK_y, pK_vx, pK_vy) to the right-hand side
of its time derivative, as an expression in those variable names. DATASET is the folder the model was fitted on
(data.npz with t and U, meta.json with the variable order); it defaults to dataset_path in RESULT_JSON.

The model is integrated once, open loop, from the first sample of the dataset over the dataset's own times;
it is never re-initialised from the data. Sample i is drawn on video frame i. If the integration blows up or
runs out of wall-clock time, the samples before the failure are kept and the rest are left empty.

The simulated trajectory is written to --sim-csv (default TRACKS_CSV with "_tracks" replaced by "_simulated")
in the tracks layout. The output video defaults to VIDEO with "-dynamics" added before the extension and has
no audio. Tracked points are hollow circles, simulated points are filled dots with a cross; both have a short
trail. The simulated-versus-tracked error is printed to stderr.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import sympy as sp
from scipy.integrate import DOP853

TRACKED = (255, 255, 0)  # BGR, cyan
SIMULATED = (0, 190, 255)  # BGR, amber


def load_model(result_json, variables):
    """The model's right-hand side as f(t, state) -> d(state)/dt, in the order of `variables`."""
    rhs = json.loads(Path(result_json).read_text())["submitted"]["rhs"]
    missing = [v for v in variables if v not in rhs]
    if missing:
        sys.exit(f"model has no equation for: {', '.join(missing)}")
    symbols = {v: sp.Symbol(v) for v in variables}
    symbols["t"] = sp.Symbol("t")
    exprs = [sp.sympify(rhs[v], locals=symbols) for v in variables]
    unknown = set().union(*(e.free_symbols for e in exprs)) - set(symbols.values())
    if unknown:
        sys.exit(f"model uses symbols that are not state variables: {sorted(map(str, unknown))}")
    fn = sp.lambdify([symbols["t"], *[symbols[v] for v in variables]], exprs, "numpy")
    return lambda t, y: np.array(fn(t, *y), dtype=float)


def simulate(f, y0, t, max_seconds=60.0, bound=1e6):
    """Integrate dy/dt = f(t, y) from y0, sampled at t. Returns (Y, message); message is None on success.

    Rows of Y after a failure (non-finite state, |state| above `bound`, solver failure, or more than
    `max_seconds` of wall-clock time) are NaN. Row 0 is y0 itself.
    """
    Y = np.full((len(t), len(y0)), np.nan)
    Y[0] = y0
    done, message, started = 1, None, time.monotonic()
    with np.errstate(all="ignore"):
        solver = DOP853(f, t[0], np.asarray(y0, dtype=float), t[-1], rtol=1e-9, atol=1e-9)
        while done < len(t):
            if time.monotonic() - started > max_seconds:
                message = f"stopped after {max_seconds:g} s of wall-clock time"
            elif solver.status == "failed" or solver.step() is not None and solver.status == "failed":
                message = "the solver failed (step size became too small)"
            elif not np.all(np.isfinite(solver.y)) or np.abs(solver.y).max() > bound:
                message = f"the state blew up (non-finite or beyond {bound:g})"
            if message:
                break
            upto = np.searchsorted(t, solver.t, side="right")
            if upto > done:
                Y[done:upto] = solver.dense_output()(t[done:upto]).T
                done = upto
    if message:
        message += f" at t = {solver.t:.3f} s; samples 0..{done - 1} (t <= {t[done - 1]:.3f} s) are valid"
    return Y, message


def error_report(names, tracked, sim, t, at_times):
    """Text table of simulated-versus-tracked error. tracked and sim are (n, points, 2) in pixels."""
    ok = np.isfinite(sim).all(axis=(1, 2)) & np.isfinite(tracked).all(axis=(1, 2))
    lines = [f"simulated vs tracked over {ok.sum()} of {len(t)} frames, pixels "
             "(rel = RMS error / standard deviation of the tracked coordinate)"]
    if not ok.any():
        return lines[0] + "\n  no frame has both a tracked and a simulated position"
    err = sim - tracked
    dist = np.linalg.norm(err, axis=2)
    lines.append(f"  {'point':6}{'dist RMS':>9}{'dist max':>9}  {'x RMS':>7}{'x max':>7}{'x sd':>7}{'x rel':>7}"
                 f"  {'y RMS':>7}{'y max':>7}{'y sd':>7}{'y rel':>7}")
    for k, name in enumerate(names):
        row = f"  {name:6}{np.sqrt((dist[ok, k] ** 2).mean()):9.2f}{dist[ok, k].max():9.2f}"
        for a in range(2):
            rms, sd = np.sqrt((err[ok, k, a] ** 2).mean()), tracked[ok, k, a].std()
            row += f"  {rms:7.2f}{np.abs(err[ok, k, a]).max():7.2f}{sd:7.2f}{rms / sd:7.2f}"
        lines.append(row)
    lines.append("  error at single frames, as dx, dy (distance):")
    for want in [*at_times, t[-1]]:
        i = int(np.abs(t - want).argmin())
        cells = [f"{n} {err[i, k, 0]:+.1f}, {err[i, k, 1]:+.1f} ({dist[i, k]:.1f})" for k, n in enumerate(names)]
        lines.append(f"  t = {t[i]:6.3f} s (frame {i}): " + ";  ".join(cells))
    return "\n".join(lines)


def fixed(points):
    """Sub-pixel points for OpenCV drawing calls with shift=4."""
    return (np.asarray(points, dtype=float) * 16).round().astype(np.int32)


def draw_trail(frame, seg, colour):
    seg = seg[np.isfinite(seg).all(axis=1)]
    if len(seg) > 1:
        cv2.polylines(frame, [fixed(np.clip(seg, -1e4, 1e4))], False, colour, 1, cv2.LINE_AA, shift=4)


def draw_tracked(frame, xy, colour=TRACKED):
    cv2.circle(frame, tuple(fixed(xy)), 9 * 16, colour, 2, cv2.LINE_AA, shift=4)


def draw_simulated(frame, xy, colour=SIMULATED):
    x, y = xy
    for dx, dy in ((7, 0), (0, 7)):
        cv2.line(frame, tuple(fixed((x - dx, y - dy))), tuple(fixed((x + dx, y + dy))), colour, 1, cv2.LINE_AA,
                 shift=4)
    cv2.circle(frame, tuple(fixed(xy)), 3 * 16, colour, -1, cv2.LINE_AA, shift=4)


def draw_legend(frame, now, sim_ok):
    font, x, y = cv2.FONT_HERSHEY_SIMPLEX, 16, 16
    panel = frame[y:y + 84, x:x + 250]
    panel[:] = (panel * 0.45).astype(np.uint8)  # darken, do not hide, what is underneath
    draw_tracked(frame, (x + 20, y + 20))
    cv2.putText(frame, "tracked", (x + 42, y + 26), font, 0.55, TRACKED, 1, cv2.LINE_AA)
    draw_simulated(frame, (x + 20, y + 46))
    label = "model (open loop)" if sim_ok else "model (failed)"
    cv2.putText(frame, label, (x + 42, y + 52), font, 0.55, SIMULATED, 1, cv2.LINE_AA)
    cv2.putText(frame, f"t = {now:5.2f} s", (x + 10, y + 76), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video", type=Path)
    ap.add_argument("tracks", type=Path)
    ap.add_argument("result", type=Path, help="result.json holding submitted -> rhs")
    ap.add_argument("-o", "--output", type=Path)
    ap.add_argument("--data", type=Path, help="dataset folder (default: dataset_path in the result file)")
    ap.add_argument("--sim-csv", type=Path, help="where to write the simulated trajectory")
    ap.add_argument("--tail", type=int, default=25, help="frames of trajectory drawn behind each point (default 25)")
    ap.add_argument("--link", nargs="*", default=[], metavar="A:B",
                    help="draw a thin line between simulated points A and B, e.g. p3:p1")
    ap.add_argument("--report-times", type=float, nargs="*", default=[1.0, 5.0, 10.0],
                    help="times (s) at which the error is listed, besides the last frame")
    ap.add_argument("--max-seconds", type=float, default=60.0, help="wall-clock limit for the integration")
    args = ap.parse_args(argv)
    output = args.output or args.video.with_name(f"{args.video.stem}-dynamics{args.video.suffix}")
    sim_csv = args.sim_csv or args.tracks.with_name(args.tracks.name.replace("_tracks", "_simulated"))
    if sim_csv == args.tracks:
        sys.exit("pass --sim-csv: the default would overwrite the tracks file")

    data_dir = args.data or Path(json.loads(args.result.read_text())["dataset_path"])
    variables = json.loads((data_dir / "meta.json").read_text())["variables"]
    data = np.load(data_dir / "data.npz")
    t, U = np.asarray(data["t"], dtype=float), np.asarray(data["U"], dtype=float)[0]

    table = pd.read_csv(args.tracks).set_index("frame")
    names = [c[:-2] for c in table.columns if c.endswith("_x")]
    absent = [f"{n}_{a}" for n in names for a in "xy" if f"{n}_{a}" not in variables]
    if absent:
        sys.exit(f"the dataset has no variable for: {', '.join(absent)}")
    links = [tuple(names.index(p) for p in link.split(":")) for link in args.link]

    Y, failure = simulate(load_model(args.result, variables), U[0], t, args.max_seconds)
    print(f"simulation: {failure or f'ran to the end (t = {t[-1]:.3f} s, {len(t)} samples)'}", file=sys.stderr)
    position_cols = [f"{n}_{a}" for n in names for a in "xy"]
    sim_table = pd.DataFrame(Y[:, [variables.index(c) for c in position_cols]], columns=position_cols)
    sim_table.insert(0, "time_s", t)
    sim_table.insert(0, "frame", np.arange(len(t)))
    sim_table.to_csv(sim_csv, index=False, float_format="%.6g")
    print(f"wrote {sim_csv}", file=sys.stderr)

    sim = sim_table[position_cols].to_numpy().reshape(len(t), len(names), 2)  # sample i belongs to frame i
    tracked = table.reindex(range(len(t)))[position_cols].to_numpy().reshape(len(t), len(names), 2)
    print(error_report(names, tracked, sim, t, args.report_times), file=sys.stderr)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = None
    for fourcc in ("avc1", "mp4v"):
        out = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if out.isOpened():
            break
    else:
        sys.exit(f"cannot open a video writer for {output}")

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i < len(t):
            lo = max(0, i - args.tail)
            sim_ok = bool(np.isfinite(sim[i]).all())
            for a, b in links:
                if sim_ok:
                    cv2.line(frame, tuple(fixed(np.clip(sim[i, a], -1e4, 1e4))),
                             tuple(fixed(np.clip(sim[i, b], -1e4, 1e4))), SIMULATED, 1, cv2.LINE_AA, shift=4)
            for k in range(len(names)):
                draw_trail(frame, tracked[lo:i + 1, k], TRACKED)
                draw_trail(frame, sim[lo:i + 1, k], SIMULATED)
                if np.isfinite(tracked[i, k]).all():
                    draw_tracked(frame, tracked[i, k])
                if np.isfinite(sim[i, k]).all():
                    draw_simulated(frame, np.clip(sim[i, k], -1e4, 1e4))
            draw_legend(frame, t[i], sim_ok)
        out.write(frame)
        i += 1
    cap.release()
    out.release()
    if i != len(t):
        print(f"warning: the video has {i} frames but the dataset has {len(t)} samples", file=sys.stderr)
    print(f"wrote {output} ({i} frames, {len(names)} points, codec {fourcc})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
