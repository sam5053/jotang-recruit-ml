# -*- coding: utf-8 -*-
"""Tiny SVG-subset renderer on top of Pillow, with per-character font fallback.

Supported: <rect> (x,y,w,h,rx,fill,stroke,stroke-width,opacity,fill-opacity),
<line>, <polyline>, <polygon>, <path> (M L C Q H V Z, absolute), <circle>,
<text> with (x,y,fs,fill,anchor,weight,rot,opacity) and plain text content.
Everything is drawn at 2x supersampling then downscaled for smooth edges.
"""
import re
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont

# order matters: first font with a real glyph wins
FONTS = [
    r"C:\Windows\Fonts\msyh.ttc",      # 中文 + 常规拉丁
    r"C:\Windows\Fonts\msyhbd.ttc",    # 中文粗体
    r"C:\Windows\Fonts\cambria.ttc",   # 数学符号、下标、希腊字母
    r"C:\Windows\Fonts\seguisym.ttf",  # 剩余符号
]
NOTDEF = "\ue000"
S = 2
_fcache = {}
_gcache = {}


def _font(path, size):
    key = (path, int(size))
    if key not in _fcache:
        _fcache[key] = ImageFont.truetype(path, max(6, int(size)))
    return _fcache[key]


def _sig(f, ch):
    m = f.getmask(ch)
    return (m.size, bytes(m))


def _has_glyph(path, ch, size=40):
    if ch.isspace():
        return True
    key = (path, ch)
    if key not in _gcache:
        f = _font(path, size)
        try:
            nd = _gcache.get((path, "NOTDEF"))
            if nd is None:
                nd = _sig(f, NOTDEF)
                _gcache[(path, "NOTDEF")] = nd
            _gcache[key] = _sig(f, ch) != nd
        except Exception:
            _gcache[key] = False
    return _gcache[key]


def pick_font(ch, size, bold):
    for p in FONTS:
        if _has_glyph(p, ch):
            return p
    return FONTS[0]


def runs(text, size, bold):
    """group consecutive chars that share a font"""
    out = []
    for ch in text:
        p = pick_font(ch, size, bold)
        if out and out[-1][0] == p:
            out[-1][1].append(ch)
        else:
            out.append([p, [ch]])
    return [(p, "".join(cs)) for p, cs in out]


def text_width(text, size, bold=False):
    w = 0.0
    for p, s in runs(text, size, bold):
        w += _font(p, size).getlength(s)
    return w


def P(v):
    return float(v) * S


def color(c):
    if c is None or c == "none":
        return None
    c = c.strip()
    if c.startswith("#"):
        c = c[1:]
        if len(c) == 3:
            c = "".join(ch * 2 for ch in c)
        return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))
    named = {"white": (255, 255, 255), "black": (0, 0, 0)}
    return named.get(c, (0, 0, 0))


def parse_path(d):
    toks = re.findall(r"[MLCQHVZmlcqhvz]|-?\d*\.?\d+(?:e-?\d+)?", d)
    segs, cur = [], None
    i, cmd = 0, None
    while i < len(toks):
        t = toks[i]
        if t.upper() in "MLCQHVZ":
            cmd = t.upper()
            i += 1
            continue
        if cmd in ("M", "L"):
            x, y = float(toks[i]), float(toks[i + 1]); i += 2
            segs.append((cmd, [(x, y)])); cur = (x, y)
        elif cmd == "C":
            pts = [(float(toks[i + 2 * k]), float(toks[i + 2 * k + 1])) for k in range(3)]
            i += 6
            segs.append(("C", pts)); cur = pts[-1]
        elif cmd == "Q":
            pts = [(float(toks[i + 2 * k]), float(toks[i + 2 * k + 1])) for k in range(2)]
            i += 4
            segs.append(("Q", pts)); cur = pts[-1]
        elif cmd == "H":
            x = float(toks[i]); i += 1
            segs.append(("L", [(x, cur[1])])); cur = (x, cur[1])
        elif cmd == "V":
            y = float(toks[i]); i += 1
            segs.append(("L", [(cur[0], y)])); cur = (cur[0], y)
        elif cmd == "Z":
            segs.append(("Z", [])); i += 1
        else:
            i += 1
    return segs


def flatten(segs, n=24):
    pts, cur, start = [], None, None
    for cmd, p in segs:
        if cmd == "M":
            cur = p[0]; start = cur; pts.append(cur)
        elif cmd == "L":
            pts.append(p[0]); cur = p[0]
        elif cmd == "C":
            p0 = cur
            for k in range(1, n + 1):
                t = k / n; mt = 1 - t
                x = (mt ** 3 * p0[0] + 3 * mt * mt * t * p[0][0]
                     + 3 * mt * t * t * p[1][0] + t ** 3 * p[2][0])
                y = (mt ** 3 * p0[1] + 3 * mt * mt * t * p[0][1]
                     + 3 * mt * t * t * p[1][1] + t ** 3 * p[2][1])
                pts.append((x, y))
            cur = p[2]
        elif cmd == "Q":
            p0 = cur
            for k in range(1, n + 1):
                t = k / n; mt = 1 - t
                x = mt * mt * p0[0] + 2 * mt * t * p[0][0] + t * t * p[1][0]
                y = mt * mt * p0[1] + 2 * mt * t * p[0][1] + t * t * p[1][1]
                pts.append((x, y))
            cur = p[1]
        elif cmd == "Z":
            if start:
                pts.append(start)
    return pts


def _draw_line(dr, x, y, s, fs, fill, anchor, bold, opacity=1.0):
    parts = runs(s, fs, bold)
    total = sum(_font(p, fs).getlength(t) for p, t in parts)
    if anchor == "middle":
        cx = x - total / 2
    elif anchor == "end":
        cx = x - total
    else:
        cx = x
    for p, t in parts:
        f = _font(p, fs)
        dr.text((cx, y), t, font=f, fill=fill + (int(255 * opacity),), anchor="ls")
        cx += f.getlength(t)
    return total


def render(svg_text, out_png, width, height, bg="#ffffff"):
    root = ET.fromstring(svg_text)
    W, H = int(P(width)), int(P(height))
    img = Image.new("RGB", (W, H), color(bg))
    dr = ImageDraw.Draw(img)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    for el in root:
        tag = el.tag.split("}")[-1]
        a = el.attrib
        op = float(a.get("opacity", 1))
        if tag == "rect":
            x, y = P(a.get("x", 0)), P(a.get("y", 0))
            w, h = P(a["width"]), P(a["height"])
            rx = P(a.get("rx", a.get("ry", 0)))
            fill = color(a.get("fill", "none"))
            stroke = color(a.get("stroke", "none"))
            fo = float(a.get("fill-opacity", 1))
            sw = max(1, int(P(a.get("stroke-width", 1.5))))
            if fill and (fo < 1 or op < 1):
                layer = Image.new("RGBA", (max(1, int(w)), max(1, int(h))), (0, 0, 0, 0))
                ld = ImageDraw.Draw(layer)
                ld.rounded_rectangle([0, 0, int(w) - 1, int(h) - 1], radius=int(rx),
                                     fill=fill + (int(255 * fo * op),))
                img.paste(layer, (int(x), int(y)), layer)
                if stroke:
                    dr.rounded_rectangle([x, y, x + w, y + h], radius=rx,
                                         outline=stroke, width=sw)
            else:
                dr.rounded_rectangle([x, y, x + w, y + h], radius=rx, fill=fill,
                                     outline=stroke, width=sw)
        elif tag == "line":
            dr.line([P(a["x1"]), P(a["y1"]), P(a["x2"]), P(a["y2"])],
                    fill=color(a.get("stroke", "#000")),
                    width=max(1, int(P(a.get("stroke-width", 1.5)))))
        elif tag == "polyline":
            pts = [tuple(map(float, p.split(","))) for p in a["points"].split()]
            dr.line([(P(x), P(y)) for x, y in pts], fill=color(a.get("stroke", "#000")),
                    width=max(1, int(P(a.get("stroke-width", 1.5)))), joint="curve")
        elif tag == "polygon":
            pts = [tuple(map(float, p.split(","))) for p in a["points"].split()]
            dr.polygon([(P(x), P(y)) for x, y in pts], fill=color(a.get("fill", "#000")),
                       outline=color(a.get("stroke", "none")))
        elif tag == "circle":
            cx, cy, r = P(a["cx"]), P(a["cy"]), P(a["r"])
            dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color(a.get("fill", "none")),
                       outline=color(a.get("stroke", "#000")),
                       width=max(1, int(P(a.get("stroke-width", 1.5)))))
        elif tag == "path":
            pts = flatten(parse_path(a["d"]))
            if a.get("fill", "none") != "none":
                dr.polygon([(P(x), P(y)) for x, y in pts], fill=color(a.get("fill")))
            if a.get("stroke", "none") != "none":
                dr.line([(P(x), P(y)) for x, y in pts], fill=color(a.get("stroke")),
                        width=max(1, int(P(a.get("stroke-width", 1.5)))), joint="curve")
        elif tag == "text":
            x, y = P(a["x"]), P(a["y"])
            fs = P(float(a.get("font-size", 16)))
            bold = a.get("font-weight", "normal") in ("bold", "700")
            fill = color(a.get("fill", "#111111"))
            anchor = a.get("text-anchor", "start")
            rot = float(a.get("rot", 0))
            lines = a.get("data-text", "").split("\n")
            lh = fs * 1.34
            if not rot:
                for k, ln in enumerate(lines):
                    _draw_line(dr, x, y + k * lh, ln, fs, fill, anchor, bold, 1)
            else:
                total = lh * (len(lines) - 1)
                bw = max([text_width(ln, fs, bold) for ln in lines] + [1])
                bh = total + fs * 1.35
                pad = int(fs * 0.35)
                tile = Image.new("RGBA", (int(bw + pad * 2), int(bh + pad * 2)), (0, 0, 0, 0))
                td = ImageDraw.Draw(tile)
                for k, ln in enumerate(lines):
                    _draw_line(td, pad, pad + fs + k * lh, ln, fs, fill, anchor, bold)
                tile = tile.rotate(rot, expand=True, resample=Image.BICUBIC)
                img.paste(tile, (int(x - tile.width / 2), int(y - tile.height / 2)), tile)

    img = img.resize((width, height), Image.LANCZOS)
    img.save(out_png)
    return out_png
