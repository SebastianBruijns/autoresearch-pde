"""Draw tracks from a tracks CSV onto the original video.

    python tracker-dusttrack/overlay_tracks.py VIDEO TRACKS_CSV [-o OUT] [--tail 25] [--stabilised]

TRACKS_CSV has the columns frame, time_s, p1_x, p1_y, ... in original-video pixels. The output defaults to
VIDEO with "-dust" added before the extension, next to VIDEO. The output has no audio.

Tracks made with camera stabilisation (track_video.py without --no-stabilise, at --scale 1.0) are in the
stabilised coordinates, not the original frame's. Pass --stabilised for those: the camera motion is estimated
again, exactly as track_video.py does it, and each point is moved back into its original frame.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def camera_transforms(video):
    """Per-frame 2x3 transforms (stabilised coords -> frame coords), as computed by track_video.py."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("dust_track_video", Path(__file__).with_name("track_video.py"))
    dust = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dust)
    frames, _, _ = dust.base.read_frames(video, 0.0, None, 1.0)
    _, transforms = dust.stabilise(frames, lambda msg: print(msg, file=sys.stderr))
    return transforms


def colours(n):
    hsv = np.array([[[int(180 * i / max(n, 1)), 255, 255] for i in range(n)]], dtype=np.uint8)
    return [tuple(int(c) for c in bgr) for bgr in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video", type=Path)
    ap.add_argument("tracks", type=Path)
    ap.add_argument("-o", "--output", type=Path)
    ap.add_argument("--tail", type=int, default=25, help="frames of trajectory drawn behind each point (default 25)")
    ap.add_argument("--stabilised", action="store_true", help="tracks are in camera-stabilised coordinates")
    args = ap.parse_args(argv)
    output = args.output or args.video.with_name(f"{args.video.stem}-dust{args.video.suffix}")

    table = pd.read_csv(args.tracks).set_index("frame")
    names = [c[:-2] for c in table.columns if c.endswith("_x")]
    cols = colours(len(names))
    transforms = camera_transforms(args.video) if args.stabilised else None

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
        for name, col in zip(names, cols):
            seg = table.loc[max(0, i - args.tail):i, [f"{name}_x", f"{name}_y"]].dropna().to_numpy()
            if i not in table.index or np.isnan(table.at[i, f"{name}_x"]):
                continue
            if transforms is not None:  # move the point and its tail into this frame's coordinates
                seg = seg @ transforms[i][:, :2].T + transforms[i][:, 2]
            cv2.polylines(frame, [(seg * 16).round().astype(np.int32)], False, col, 2, cv2.LINE_AA, shift=4)
            x, y = seg[-1]
            cv2.circle(frame, (int(round(x * 16)), int(round(y * 16))), 10 * 16, col, 2, cv2.LINE_AA, shift=4)
            cv2.putText(frame, name, (int(x) + 14, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        out.write(frame)
        i += 1
    cap.release()
    out.release()
    print(f"wrote {output} ({i} frames, {len(names)} points, codec {fourcc})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
