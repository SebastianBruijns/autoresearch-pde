"""Generate the Lorenz logo: an abstract vortex, five swept rings narrowing down into a bending tail.

Each ring is the front of a funnel cross-section drawn as a comet stroke: a round head on the left, tapering as it
sweeps right, so the stack reads as spinning. Run: python demo/brand/make_logo.py
"""
import subprocess
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent
INK, ACCENT, VIOLET = "#1d1f22", "#2a78d6", "#4a3aa7"   # theme colours (.streamlit/config.toml)
FONT = "https://github.com/google/fonts/raw/main/ofl/sourceserif4/SourceSerif4%5Bopsz,wght%5D.ttf"


def stroke(x, y, w):
    """Closed outline of a centreline (x, y) with per-point width w and round caps."""
    dx, dy = np.gradient(x), np.gradient(y)
    L = np.hypot(dx, dy)
    nx, ny = -dy / L, dx / L
    a = np.linspace(0, np.pi, 16)[1:-1]
    b0, b1 = np.arctan2(ny[0], nx[0]), np.arctan2(ny[-1], nx[-1])
    end = np.c_[x[-1] + w[-1] / 2 * np.cos(b1 - a), y[-1] + w[-1] / 2 * np.sin(b1 - a)]
    start = np.c_[x[0] + w[0] / 2 * np.cos(b0 + np.pi - a), y[0] + w[0] / 2 * np.sin(b0 + np.pi - a)]
    return np.vstack([np.c_[x + nx * w / 2, y + ny * w / 2], end, np.c_[x - nx * w / 2, y - ny * w / 2][::-1], start])


def vortex(k=5, wmax=0.17, gap=0.56, tilt=0.3, bend=0.5, rmin=0.2, m=240):
    """List of ring outlines (y up). Rings shrink and drift right as they go down; spacing tightens slightly."""
    rings = []
    for i in range(k):
        f = i / (k - 1)
        R = rmin + (1 - rmin) * (1 - f) ** 1.2
        cx, cy = bend * f ** 1.8, -gap * (i - 0.06 * i * (i - 1))
        u = np.linspace(0, 1, m)
        th = np.pi * (1 - u)                                   # left to right along the front of the ring
        x, y = cx + R * np.cos(th), cy - tilt * R * np.sin(th)
        rings.append(stroke(x, y, (wmax * (1 - u) ** 0.7 + 0.012) * (1 - 0.25 * f)))
    return rings


def path_d(rings, k=1.0, dx=0.0, dy=0.0):
    return "".join("M" + " L".join(f"{(px - dx) * k:.4f},{(dy - py) * k:.4f}" for px, py in r) + " Z" for r in rings)


def gradient(id_):
    return (f"<defs><linearGradient id='{id_}' x1='0' y1='0' x2='0' y2='1'><stop offset='0' stop-color='{ACCENT}'/>"
            f"<stop offset='1' stop-color='{VIOLET}'/></linearGradient></defs>")


def mark_svg(rings, color=ACCENT, pad=0.11, bg=None, size=512, grad=False):
    allp = np.vstack(rings)
    (x0, y0), (x1, y1) = allp.min(0), allp.max(0)
    s = max(x1 - x0, y1 - y0) * (1 + 2 * pad)
    cx, cy = (x0 + x1) / 2, -(y0 + y1) / 2
    rect = (f"<rect x='{cx - s / 2:.4f}' y='{cy - s / 2:.4f}' width='{s:.4f}' height='{s:.4f}' rx='{s * .22:.4f}' "
            f"fill='{bg}'/>") if bg else ""
    fill = "url(#g)" if grad else color
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='{cx - s / 2:.4f} {cy - s / 2:.4f} {s:.4f} {s:.4f}' "
            f"width='{size}' height='{size}'>{gradient('g') if grad else ''}{rect}"
            f"<path d='{path_d(rings)}' fill='{fill}'/></svg>")


def text_path(text, font_file, wght=600, opsz=32):
    """SVG path of `text` in Source Serif 4 (outlined, so the logo needs no web font). Units: em = 1, baseline y = 0."""
    from fontTools.pens.svgPathPen import SVGPathPen
    from fontTools.pens.transformPen import TransformPen
    from fontTools.ttLib import TTFont
    from fontTools.varLib.instancer import instantiateVariableFont
    f = instantiateVariableFont(TTFont(font_file), {"wght": wght, "opsz": opsz})
    upm, cmap, gs = f["head"].unitsPerEm, f.getBestCmap(), f.getGlyphSet()
    pen, x = SVGPathPen(gs), 0
    for ch in text:
        g = cmap[ord(ch)]
        gs[g].draw(TransformPen(pen, (1 / upm, 0, 0, -1 / upm, x, 0)))
        x += gs[g].width / upm
    return pen.getCommands(), x


def lockup_svg(rings, name, font_file, height=96):
    """Horizontal logo: the mark spans cap height (0.66 em) plus a little, then the name."""
    d, adv = text_path(name, font_file)
    allp = np.vstack(rings)
    (x0, y0), (x1, y1) = allp.min(0), allp.max(0)
    k = 0.8 / (y1 - y0)                                       # mark height in em
    mark = path_d(rings, k, x0, y1)                           # top at y = 0; shifted up to sit on the cap height
    tx = (x1 - x0) * k + 0.2
    top, h = -0.86, 1.12
    w = tx + adv + 0.04
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='-0.02 {top} {w:.4f} {h}' height='{height}' "
            f"width='{height * w / h:.0f}'><path transform='translate(0 -0.73)' d='{mark}' fill='{ACCENT}'/>"
            f"<path transform='translate({tx:.4f} 0)' d='{d}' fill='{INK}'/></svg>")


if __name__ == "__main__":
    rings = vortex()
    font = Path("/tmp/SourceSerif4.ttf")
    if not font.exists():
        subprocess.run(["curl", "-sSL", "-o", str(font), FONT], check=True)
    (OUT / "mark.svg").write_text(mark_svg(rings))
    (OUT / "mark-gradient.svg").write_text(mark_svg(rings, grad=True))
    (OUT / "icon.svg").write_text(mark_svg(vortex(wmax=0.22), "#fff", pad=0.2, bg=ACCENT))
    (OUT / "logo-lorenz.svg").write_text(lockup_svg(vortex(wmax=0.23), "Lorenz", font))   # heavier beside text
    subprocess.run(["rsvg-convert", "-w", "64", str(OUT / "icon.svg"), "-o", str(OUT / "icon-64.png")], check=True)
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.svg"))))
