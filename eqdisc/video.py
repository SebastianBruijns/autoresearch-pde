"""Turn a video file into something Claude can read, and send it to Modal.

The Claude API takes images, not video files. A video is therefore sent as its frames, in order, as JPEG image blocks.
This module does the decoding on the local machine (ffmpeg, in memory: nothing is written to disk) and packs the frames
into a small picklable `VideoPayload` that can be passed to a Modal function as an argument.

    from eqdisc.video import load_video, ask
    v = load_video("clip.mov", max_frames=100, width=768)     # local: decode + pick frames
    result = ask(v, "gimme pde", model="claude-opus-5-5")      # anywhere (e.g. inside a Modal container)

API limits that `load_video` respects by default: at most 100 images per request, long edge <= 2000 px when more than
20 images are sent (frames are scaled to `width`, default 768, which also keeps the token cost down), and well under
the 32 MB request size. `ask` makes one streamed call and keeps everything the API returns: the thinking (the API
returns a summary of the model's reasoning, not the raw chain), the text, the usage and the full raw response.
"""
import base64
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

MAX_IMAGES = 100            # images per API request
MAX_EDGE_MANY = 2000        # long-edge limit (px) when more than 20 images are sent


@dataclass
class VideoPayload:
    """Frames of one video, ready to send. Picklable (bytes + numbers), so it can be a Modal function argument."""
    jpegs: list                       # JPEG bytes, in time order
    times: list                       # timestamp of each frame (s from the start of the video)
    fps: float                        # frame rate of the source video
    size: tuple                       # (width, height) of the sent frames
    source: dict = field(default_factory=dict)   # probe info of the source file (name, size, fps, frame count, duration)

    def content(self, prompt=None, timestamps=False):
        """Anthropic message content: the frames as image blocks, then `prompt` (if any) as a text block.
        timestamps=True puts a short "t = 0.40 s" text block before each frame (off by default: adds context)."""
        blocks = []
        times = self.times or [None] * len(self.jpegs)
        for j, t in zip(self.jpegs, times):
            if timestamps and t is not None:
                blocks.append({"type": "text", "text": f"t = {t:.2f} s"})
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                       "data": base64.b64encode(j).decode()}})
        if prompt:
            blocks.append({"type": "text", "text": prompt})
        return blocks

    def summary(self):
        return {"n_frames": len(self.jpegs), "fps": self.fps, "size": list(self.size),
                "t_first": self.times[0] if self.times else None, "t_last": self.times[-1] if self.times else None,
                "bytes": sum(len(j) for j in self.jpegs), "source": self.source}


def probe(path):
    """(width, height, fps, n_frames, duration) of the first video stream; ffprobe applies the rotation tag."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_format",
                          "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    s = info["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    w, h = int(s["width"]), int(s["height"])
    rot = next((abs(int(d.get("rotation", 0))) for d in s.get("side_data_list", []) if "rotation" in d),
               abs(int(s.get("tags", {}).get("rotate", 0))))
    if rot in (90, 270):                              # ffmpeg auto-rotates on decode
        w, h = h, w
    dur = float(s.get("duration") or info.get("format", {}).get("duration") or 0)
    n = int(s.get("nb_frames") or round(dur * fps))
    return w, h, fps, n, dur


def _scaled(w, h, width):
    width = min(width, w)
    return width, int(round(h * width / w / 2)) * 2


def decode_jpegs(path, width=768, quality=4, start=None, end=None):
    """Every frame (optionally between start/end seconds) as JPEG bytes, scaled to `width` px (aspect kept)."""
    w, h, fps, _, _ = probe(path)
    sw, sh = _scaled(w, h, width)
    cmd = ["ffmpeg", "-v", "error"]
    if start is not None:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(path)]
    if end is not None:
        cmd += ["-t", str(end - (start or 0))]
    cmd += ["-vf", f"scale={sw}:{sh}", "-c:v", "mjpeg", "-q:v", str(quality), "-f", "image2pipe", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames, i = [], 0
    while True:                                       # split the MJPEG stream on SOI/EOI markers
        a = raw.find(b"\xff\xd8", i)
        b = raw.find(b"\xff\xd9", a)
        if a < 0 or b < 0:
            break
        frames.append(raw[a:b + 2])
        i = b + 2
    return frames, fps, (sw, sh)


def decode_rgb(path, width=96):
    """All frames as a uint8 array (n, h, w, 3) scaled to `width` px, plus fps. For turning a video into fields."""
    import numpy as np
    w, h, fps, _, _ = probe(path)
    sw, sh = _scaled(w, h, width)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"scale={sw}:{sh}", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, sh, sw, 3), fps


def load_video(path, max_frames=MAX_IMAGES, width=768, quality=4, start=None, end=None):
    """Decode `path` and keep at most `max_frames` evenly spaced frames (all of them if the clip is short enough)."""
    path = Path(path).expanduser().resolve()
    max_frames = min(max_frames, MAX_IMAGES)
    if max_frames > 20:
        width = min(width, MAX_EDGE_MANY)
    w, h, fps, n, dur = probe(path)
    frames, fps, size = decode_jpegs(path, width, quality, start, end)
    if max(size) > MAX_EDGE_MANY and max_frames > 20:
        raise ValueError(f"frames {size} exceed {MAX_EDGE_MANY}px; lower width")
    idx = list(range(len(frames)))
    if len(idx) > max_frames:                         # evenly spaced, first and last frame included
        idx = sorted({round(k * (len(frames) - 1) / (max_frames - 1)) for k in range(max_frames)})
    t0 = start or 0.0
    return VideoPayload(jpegs=[frames[i] for i in idx], times=[round(t0 + i / fps, 4) for i in idx], fps=fps,
                        size=size, source={"name": path.name, "width": w, "height": h, "fps": fps, "n_frames": n,
                                           "duration_s": dur, "frame_index": idx})


def _jsonable(msg):
    try:
        return json.loads(msg.model_dump_json())
    except Exception:  # noqa: BLE001
        return str(msg)


def ask(video, prompt, model="claude-opus-5-5", effort="high", system=None, max_tokens=64000, timestamps=False,
        client=None):
    """One streamed API call with the video's frames and `prompt`. No system prompt or tools unless given.
    Returns a result dict with the thinking/text log, usage and cost, and the full raw response."""
    import anthropic
    from eqdisc.llm import request_opts
    from eqdisc.srsd import Cost
    t0 = time.time()
    client = client or anthropic.Anthropic()
    kw = dict(model=model, max_tokens=max_tokens,
              messages=[{"role": "user", "content": video.content(prompt, timestamps)}], **request_opts(model, effort))
    if system:
        kw["system"] = system
    with client.beta.messages.stream(**kw) as s:
        msg = s.get_final_message()
    cost = Cost(model)
    cost.add(msg)
    log = []
    for b in msg.content:
        if b.type == "thinking" and (getattr(b, "thinking", "") or "").strip():
            log.append({"type": "thinking", "text": b.thinking})
        elif b.type == "redacted_thinking":
            log.append({"type": "thinking", "text": "(redacted by the API)"})
        elif b.type == "text" and b.text.strip():
            log.append({"type": "text", "text": b.text})
    answer = "\n".join(e["text"] for e in log if e["type"] == "text")
    return {"model": model, "effort": effort, "prompt": prompt, "stop": msg.stop_reason, "n_tool_calls": 0,
            "usage": cost.as_dict(), "cost_usd": cost.as_dict()["cost_usd"], "log": log, "answer": answer,
            "video": video.summary(), "wall_s": round(time.time() - t0, 1), "raw_response": _jsonable(msg)}
