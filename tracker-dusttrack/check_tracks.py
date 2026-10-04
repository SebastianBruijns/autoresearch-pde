"""Contact sheet for checking tracks by eye: zoomed crops centred on every tracked point, at frames spread
through the clip. One row per point, one column per frame.

    python tracker-dusttrack/check_tracks.py ANNOTATED.mp4 TRACKS.csv -o sheet.png [--n 8] [--frames 10 20 ...]

ANNOTATED.mp4 is the --annotated output of track_video.py (the stabilised clip the tracks refer to, with a
6 px circle already drawn at each tracked position), TRACKS.csv its CSV. In each tile the tracked position
is the tile centre, marked by four short white ticks; a correct track keeps the same physical feature
under the centre in every tile of a row. A black tile means the point has no position in that frame.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("annotated", type=Path)
    ap.add_argument("tracks", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--n", type=int, default=8, help="number of frames, evenly spread (default 8)")
    ap.add_argument("--frames", type=int, nargs="+", help="explicit frame numbers (as in the CSV) instead of --n")
    ap.add_argument("--radius", type=int, default=40, help="crop half-size in pixels (default 40)")
    ap.add_argument("--zoom", type=int, default=3)
    ap.add_argument("--scale", type=float, default=1.0, help="the --scale used for track_video.py")
    args = ap.parse_args()
    t = pd.read_csv(args.tracks)
    names = [c[:-2] for c in t.columns if c.endswith("_x")]
    wanted = args.frames or [int(v) for v in np.linspace(t.frame.iloc[0], t.frame.iloc[-1], args.n).round()]
    rows_by_frame = t.set_index("frame")
    cap = cv2.VideoCapture(str(args.annotated))
    grabbed, i = {}, int(t.frame.iloc[0])
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in wanted:
            grabbed[i] = f
        i += 1
    r, z = args.radius, args.zoom
    side = 2 * r * z
    sheet = []
    for name in names:
        tiles = []
        for fnum in wanted:
            tile = np.zeros((side, side, 3), np.uint8)
            x, y = rows_by_frame.loc[fnum, f"{name}_x"] * args.scale, rows_by_frame.loc[fnum, f"{name}_y"] * args.scale
            if fnum in grabbed and np.isfinite(x):
                M = np.array([[z, 0, z * (r - x)], [0, z, z * (r - y)]], dtype=np.float64)
                tile = cv2.warpAffine(grabbed[fnum], M, (side, side), flags=cv2.INTER_CUBIC)
                c = side // 2
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    cv2.line(tile, (c + dx * 30, c + dy * 30), (c + dx * 55, c + dy * 55), (255, 255, 255), 1)
            cv2.putText(tile, f"{name} f{fnum}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.rectangle(tile, (0, 0), (side - 1, side - 1), (90, 90, 90), 1)
            tiles.append(tile)
        sheet.append(np.hstack(tiles))
    cv2.imwrite(str(args.output), np.vstack(sheet))
    print(f"wrote {args.output}: {len(names)} points x {len(wanted)} frames", file=sys.stderr)


if __name__ == "__main__":
    main()
