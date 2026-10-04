"""Add velocity columns to a tracks CSV.

    python tracker-dusttrack/add_velocity.py TRACKS_CSV [-o OUT] [--fps FPS] [--window 11] [--drop-frame]

TRACKS_CSV has the columns frame, time_s, p1_x, p1_y, ... (as written by the track_video.py programs). For every
position column pK_x / pK_y a velocity column pK_vx / pK_vy is added, in pixels per second. The output defaults
to TRACKS_CSV with "-velocity" added before the extension.

Velocities are the derivative of a local cubic fitted over --window samples (Savitzky-Golay), which is far less
noisy than differencing neighbouring frames. Positions are left as tracked. With --fps, time_s is recomputed
as frame / FPS (the trackers round time_s, which makes the sampling look uneven). Tracks with gaps are not
supported: every position cell must be filled.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tracks", type=Path)
    ap.add_argument("-o", "--output", type=Path)
    ap.add_argument("--fps", type=float, help="recompute time_s as frame / FPS")
    ap.add_argument("--window", type=int, default=11, help="samples in the smoothing window, odd (default 11)")
    ap.add_argument("--drop-frame", action="store_true", help="leave the frame column out of the output")
    args = ap.parse_args(argv)
    output = args.output or args.tracks.with_name(f"{args.tracks.stem}-velocity{args.tracks.suffix}")

    table = pd.read_csv(args.tracks)
    if args.fps:
        table["time_s"] = table["frame"] / args.fps
    positions = [c for c in table.columns if c.endswith(("_x", "_y"))]
    if table[positions].isna().any().any():
        sys.exit("tracks have empty cells; fill or cut the gaps before computing velocities")
    dt = float(np.median(np.diff(table["time_s"])))
    for col in positions:
        name, axis = col.rsplit("_", 1)
        table[f"{name}_v{axis}"] = savgol_filter(table[col].to_numpy(), args.window, 3, deriv=1, delta=dt)
    if args.drop_frame:
        table = table.drop(columns="frame")
    table.to_csv(output, index=False, float_format="%.6f")
    print(f"wrote {output} ({len(table)} rows, {len(positions)} velocity columns, dt {dt:.6f} s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
