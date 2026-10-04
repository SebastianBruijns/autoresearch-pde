"""Generate the logo: an Euler spiral (the curve of the Fresnel integrals) drawn as Leibniz's integral sign.

The clothoid's centre reads as the long s of the integral; its two ends wind into opposite vortices. The stroke is
calligraphic: widest in the middle, thinning into the spiral cores. Run: python demo/brand/make_logo.py
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent
INK, ACCENT = "#1d1f22", "#2a78d6"     # theme text and primary colours (.streamlit/config.toml)
FONT = "https://github.com/google/fonts/raw/main/ofl/sourceserif4/SourceSerif4%5Bopsz,wght%5D.ttf"


def outline(T=2.4, stem=0.4, w_mid=0.16, w_end=0.014, slant=76, n=1400):
    """Closed polygon (x, y) of the variable-width stroke, y pointing up.

    Curvature grows linearly with arc length beyond a straight stem of half-length `stem` (two Euler spirals joined by
    a line), so the ends are true clothoid vortices while the middle stays long, like a typeset integral.
    """
    s = np.linspace(-(T + stem), T + stem, n)
    t = np.sign(s) * np.maximum(np.abs(s) - stem, 0)       # clothoid parameter: 0 along the stem
    theta = -np.pi * t ** 2 / 2                            # curvature pi*t, mirrored so the top end curls right
    ds = s[1] - s[0]
    x, y = np.cumsum(np.cos(theta)) * ds, np.cumsum(np.sin(theta)) * ds
    x, y = x - x.mean(), y - y.mean()
    a = np.deg2rad(slant)                                   # lean the stem like an italic long s
    x, y = x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a)
    dx, dy = np.gradient(x), np.gradient(y)
    L = np.hypot(dx, dy)
    nx, ny = -dy / L, dx / L
    w = w_end + (w_mid - w_end) * np.exp(-(t / (0.55 * T)) ** 2)
    w = np.minimum(w, 0.9 / (np.pi * np.abs(t) + 1e-9) * 0.5)   # never wider than the local radius of curvature
    left = np.c_[x + nx * w / 2, y + ny * w / 2]
    right = np.c_[x - nx * w / 2, y - ny * w / 2][::-1]
    ang = np.linspace(0, np.pi, 12)[1:-1]   # round caps
    def cap(i, sgn):
        c, r = np.array([x[i], y[i]]), w[i] / 2
        base = np.arctan2(ny[i], nx[i])
        return np.c_[c[0] + r * np.cos(base + sgn * ang), c[1] + r * np.sin(base + sgn * ang)]
    return np.vstack([left, cap(-1, -1), right, cap(0, -1)]) * [1, -1]   # flip to SVG coordinates (y down)


def path_d(poly):
    return "M" + " L".join(f"{px:.4f},{py:.4f}" for px, py in poly) + " Z"


def mark_svg(poly, color=INK, pad=0.12, bg=None, size=512):
    x0, y0 = poly.min(0)
    x1, y1 = poly.max(0)
    s = max(x1 - x0, y1 - y0) * (1 + 2 * pad)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    vb = f"{cx - s / 2:.4f} {cy - s / 2:.4f} {s:.4f} {s:.4f}"
    rect = (f"<rect x='{cx - s / 2:.4f}' y='{cy - s / 2:.4f}' width='{s:.4f}' height='{s:.4f}' rx='{s * .22:.4f}' "
            f"fill='{bg}'/>") if bg else ""
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='{vb}' width='{size}' height='{size}'>{rect}"
            f"<path d='{path_d(poly)}' fill='{color}'/></svg>")


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


def lockup_svg(poly, name, font_file, height=96):
    """Horizontal logo: mark, then the name; the mark spans cap height to descender, like an integral beside text."""
    d, adv = text_path(name, font_file)
    x0, y0 = poly.min(0)
    x1, y1 = poly.max(0)
    k = 1.05 / (y1 - y0)                    # mark height in em
    mx = lambda p: f"{(p[0] - x0) * k:.4f},{(p[1] - y0) * k - 0.83:.4f}"
    mark = "M" + " L".join(mx(p) for p in poly) + " Z"
    tx = (x1 - x0) * k + 0.16
    w, top, h = tx + adv + 0.04, -0.86, 1.12
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='-0.02 {top} {w:.4f} {h}' height='{height}' "
            f"width='{height * w / h:.0f}'><path d='{mark}' fill='{ACCENT}'/>"
            f"<path transform='translate({tx:.4f} 0)' d='{d}' fill='{INK}'/></svg>")


if __name__ == "__main__":
    poly = outline(w_mid=0.19, w_end=0.02)
    font = Path("/tmp/SourceSerif4.ttf")
    if not font.exists():
        subprocess.run(["curl", "-sSL", "-o", str(font), FONT], check=True)
    (OUT / "mark.svg").write_text(mark_svg(poly, ACCENT))
    (OUT / "mark-ink.svg").write_text(mark_svg(poly, INK))
    (OUT / "icon.svg").write_text(mark_svg(outline(T=2.1, w_mid=0.26, w_end=0.05), "#fff", pad=0.2, bg=ACCENT))
    for name in sys.argv[1:] or ["eqdisc", "Gottfried"]:
        (OUT / f"logo-{name.lower()}.svg").write_text(lockup_svg(poly, name, font))
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.svg"))))
