"""Mk33 brand assets: logo, README banner and GitHub social card.

The design is a contour map of a score landscape. A dotted trail of rounds climbs to
the summit, marked "ROUND 33" in amber.

    pip install numpy matplotlib fonttools brotli cairosvg
    python assets/make_assets.py --fetch-fonts      # once: downloads the two typefaces via npm
    python assets/make_assets.py                    # writes assets/banner.svg, logo.svg, social-preview.png

Typefaces: Space Grotesk (400, 700) and JetBrains Mono (500), both SIL Open Font License,
from the @fontsource npm packages. Text is converted to outlines, so the SVGs render the
same everywhere; GitHub shows README images without web fonts. Change the wording in
make_summit() and rerun.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

OUT = Path(__file__).resolve().parent
FONTS = OUT / ".fonts"  # git-ignored
NEEDED = {"space-grotesk": ["400", "700"], "jetbrains-mono": ["500"]}


def fetch_fonts() -> None:
    """Download the typefaces from npm and convert the woff2 files to TTF."""
    import subprocess
    import tarfile
    import tempfile

    FONTS.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for pkg, weights in NEEDED.items():
            name = subprocess.run(["npm", "pack", "-q", f"@fontsource/{pkg}"], cwd=tmp, check=True,
                                  capture_output=True, text=True).stdout.strip().splitlines()[-1]
            with tarfile.open(Path(tmp) / name) as t:
                t.extractall(Path(tmp) / pkg, filter="data")
            for w in weights:
                src = Path(tmp) / pkg / "package" / "files" / f"{pkg}-latin-{w}-normal.woff2"
                f = TTFont(src)
                f.flavor = None
                f.save(FONTS / f"{pkg}-latin-{w}-normal.ttf")
    print("fonts saved to", FONTS)


if "--fetch-fonts" in sys.argv:
    fetch_fonts()
    sys.exit(0)
missing = [f"{p}-latin-{w}-normal.ttf" for p, ws in NEEDED.items() for w in ws
           if not (FONTS / f"{p}-latin-{w}-normal.ttf").exists()]
if missing:
    sys.exit(f"missing fonts {missing}; run: python {Path(__file__).name} --fetch-fonts")

_cache: dict[str, TTFont] = {}


def font(name: str) -> TTFont:
    if name not in _cache:
        _cache[name] = TTFont(FONTS / name)
    return _cache[name]


def text(s: str, x: float, y: float, size: float, face: str, fill: str, tracking: float = 0.0,
         anchor: str = "start", opacity: float | None = None) -> tuple[str, float]:
    """Outlined text as an SVG path; y is the baseline. Returns (svg, width)."""
    f = font(face)
    upm = f["head"].unitsPerEm
    cmap, gs, hmtx = f.getBestCmap(), f.getGlyphSet(), f["hmtx"]
    scale = size / upm
    names = [cmap.get(ord(ch), cmap.get(ord("?"))) for ch in s]
    adv = [hmtx[n][0] * scale + tracking for n in names]
    width = sum(adv) - (tracking if names else 0)
    x0 = x - width if anchor == "end" else x - width / 2 if anchor == "middle" else x
    pen = SVGPathPen(gs)
    cx = x0
    for n, a in zip(names, adv):
        gs[n].draw(TransformPen(pen, (scale, 0, 0, -scale, cx, y)))
        cx += a
    op = f' fill-opacity="{opacity}"' if opacity is not None else ""
    return f'<path d="{pen.getCommands()}" fill="{fill}"{op}/>', width


def svg(w: int, h: int, body: str, defs: str = "") -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            + (f"<defs>{defs}</defs>" if defs else "") + body + "</svg>\n")


def save(name: str, content: str) -> None:
    (OUT / name).write_text(content)


# ------------------------------------------------------------------ direction 1: summit
# A dark contour map of a score landscape; a trail of 33 rounds climbs to the peak.

S = dict(bg="#0c0e12", line="#252b36", line_hi="#3b4454", accent="#f0a43a", ink="#f4f2ed",
         ink2="#a3abb8", ink3="#6d7584")


def landscape(nx: int, ny: int, peak: tuple[float, float], seed: int = 3):
    rng = np.random.default_rng(seed)
    xs, ys = np.linspace(0, 1, nx), np.linspace(0, 1, ny)
    X, Y = np.meshgrid(xs, ys)
    Z = 1.2 * np.exp(-(((X - peak[0]) / 0.22) ** 2 + ((Y - peak[1]) / 0.30) ** 2))
    for _ in range(7):  # foothills
        cx, cy = rng.uniform(0, 1), rng.uniform(0, 1)
        Z += rng.uniform(0.15, 0.45) * np.exp(-(((X - cx) / rng.uniform(0.08, 0.2)) ** 2
                                                 + ((Y - cy) / rng.uniform(0.1, 0.25)) ** 2))
    return xs, ys, Z


def contours(w: float, h: float, ox: float, oy: float, peak, levels: int, seed: int = 3):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs, ys, Z = landscape(220, int(220 * h / w) or 60, peak, seed)
    fig, ax = plt.subplots()
    cs = ax.contour(xs, ys, Z, levels=levels)
    out = []
    for i, segs in enumerate(cs.allsegs):
        for seg in segs:
            if len(seg) < 3:
                continue
            pts = " ".join(f"{ox + px * w:.1f},{oy + py * h:.1f}" for px, py in seg)
            out.append((i, f"M{pts.replace(' ', ' L', 0)}"))
    plt.close(fig)
    return out, len(cs.allsegs), (xs, ys, Z)


def trail(xs, ys, Z, start, peak, n=33, seed=7):
    """A climbing path of n points (one per round) from start to the peak, with plateaus."""
    rng = random.Random(seed)
    pts = []
    for i in range(n):
        t = (i / (n - 1)) ** 0.8
        x = start[0] + (peak[0] - start[0]) * t + rng.uniform(-0.02, 0.02) * (1 - t)
        y = start[1] + (peak[1] - start[1]) * t + 0.08 * math.sin(t * 5.5) * (1 - t)
        pts.append((x, y))
    pts[-1] = peak
    return pts


def summit_scene(W, H, ox, oy, w, h, peak, levels, dot_scale=1.0, fade=True, seed=3, n_dots=33,
                 start=(0.08, 0.85)):
    segs, nlev, (xs, ys, Z) = contours(w, h, ox, oy, peak, levels, seed)
    iy, ix = np.unravel_index(np.argmax(Z), Z.shape)
    peak = (float(xs[ix]), float(ys[iy]))  # the true summit, after the foothills are added
    body = []
    for i, d in segs:
        top = i >= nlev - 3
        color = S["accent"] if top else (S["line_hi"] if i >= nlev - 7 else S["line"])
        sw = 1.6 if top else 1.1
        if dot_scale >= 2:  # logo: fewer, heavier, brighter lines so it survives favicon sizes
            color = S["accent"] if top else "#4a5466"
            sw = 7 if top else 4.5
        body.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" '
                    f'stroke-linejoin="round" stroke-linecap="round"/>')
    pts = trail(xs, ys, Z, start, peak, n=n_dots)
    P = [(ox + x * w, oy + y * h) for x, y in pts]
    body.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in P)
                + f'" fill="none" stroke="{S["ink"]}" stroke-opacity="0.55" stroke-width="{1.2 * dot_scale}" '
                f'stroke-dasharray="{3 * dot_scale} {4 * dot_scale}"/>')
    for x, y in P[:-1]:
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{2.1 * dot_scale}" fill="{S["ink"]}" fill-opacity="0.8"/>')
    px, py = P[-1]
    if dot_scale < 2:
        body.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{11 * dot_scale}" fill="none" stroke="{S["accent"]}" '
                    f'stroke-width="{1.5 * dot_scale}"/>')
    body.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{(5 if dot_scale < 2 else 2.6) * dot_scale:.1f}" '
                f'fill="{S["accent"]}"/>')
    group = "".join(body)
    if not fade:
        return group, (px, py), ""
    defs = (f'<linearGradient id="fade" x1="0" x2="1" y1="0" y2="0">'
            f'<stop offset="0" stop-color="#fff" stop-opacity="0"/>'
            f'<stop offset="0.45" stop-color="#fff" stop-opacity="1"/></linearGradient>'
            f'<mask id="m"><rect width="{W}" height="{H}" fill="url(#fade)"/></mask>')
    return f'<g mask="url(#m)">{group}</g>', (px, py), defs


def pill_label(label, x, y, size, face, color):
    t, w = text(label, x, y, size, face, color, tracking=size * 0.13)
    pad_x, pad_y = size * 0.6, size * 0.55
    rect = (f'<rect x="{x - pad_x:.1f}" y="{y - size * 0.95 - pad_y + size * 0.2:.1f}" width="{w + 2 * pad_x:.1f}" '
            f'height="{size * 0.95 + 2 * pad_y:.1f}" rx="{size * 0.5:.1f}" fill="{S["bg"]}" stroke="{S["accent"]}" '
            f'stroke-opacity="0.5" stroke-width="1"/>')
    return rect + t


def make_summit():
    GS, MONO = "space-grotesk-latin-700-normal.ttf", "jetbrains-mono-latin-500-normal.ttf"
    GS4 = "space-grotesk-latin-400-normal.ttf"
    TAGS = "RLC  ·  RAG MEMORY  ·  KV-CACHE EVICTION  ·  TRITON"

    # banner 1280 x 360
    W, H = 1280, 360
    scene, (px, py), defs = summit_scene(W, H, 330, -10, 980, 390, (0.72, 0.42), 22)
    t1, _ = text("Mk33", 64, 196, 124, GS, S["ink"], tracking=-4)
    t2, _ = text("Long-horizon self-improving agents", 68, 238, 22, GS4, S["ink2"])
    t3, _ = text(TAGS, 68, 290, 12.5, MONO, S["ink3"], tracking=1.2)
    lab = pill_label("ROUND 33", px - 30, py - 34, 11, MONO, S["accent"])
    save("summit_banner.svg", svg(W, H, f'<rect width="{W}" height="{H}" fill="{S["bg"]}"/>' + scene
                                  + t1 + t2 + t3 + lab, defs))

    # social card 1280 x 640
    W, H = 1280, 640
    scene, (px, py), defs = summit_scene(W, H, 300, 0, 1000, 640, (0.68, 0.40), 26, start=(0.36, 0.97))
    t1, _ = text("Mk33", 72, 360, 168, GS, S["ink"], tracking=-6)
    t2, _ = text("Long-horizon self-improving agents", 78, 418, 30, GS4, S["ink2"])
    t3, _ = text(TAGS, 80, 500, 15, MONO, S["ink3"], tracking=1.4)
    t4, _ = text("github.com/dulguuntsatsralt-cell/Mk33", 80, 580, 15, MONO, S["ink3"], tracking=0.6)
    lab = pill_label("ROUND 33", px - 34, py - 40, 13, MONO, S["accent"])
    save("summit_social.svg", svg(W, H, f'<rect width="{W}" height="{H}" fill="{S["bg"]}"/>' + scene
                                  + t1 + t2 + t3 + t4 + lab, defs))

    # logo 512: contour rings around the summit, no text
    W = H = 512
    scene, _, _ = summit_scene(W, H, 0, 0, 512, 512, (0.6, 0.42), 7, dot_scale=4.2, fade=False, seed=11, n_dots=7)
    clip = f'<clipPath id="c"><rect width="512" height="512" rx="112"/></clipPath>'
    save("summit_logo.svg", svg(W, H, f'<g clip-path="url(#c)"><rect width="512" height="512" fill="{S["bg"]}"/>'
                                + scene + "</g>", clip))


make_summit()
for old, new in (("summit_banner.svg", "banner.svg"), ("summit_logo.svg", "logo.svg"),
                 ("summit_social.svg", "social-preview.svg")):
    (OUT / old).replace(OUT / new)
try:
    import cairosvg

    cairosvg.svg2png(url=str(OUT / "social-preview.svg"), write_to=str(OUT / "social-preview.png"),
                     output_width=1280, output_height=640)
    (OUT / "social-preview.svg").unlink()
except ImportError:
    print("cairosvg not installed: kept social-preview.svg; convert it to a 1280x640 PNG for GitHub")
print("wrote", sorted(p.name for p in OUT.iterdir() if p.suffix in (".svg", ".png")))
