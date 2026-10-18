# -*- coding: utf-8 -*-
"""Minimal PDF text extractor: inflate FlateDecode streams, pull text ops from
content streams in page order. Good enough for an arXiv-style PDF."""
import re, sys, zlib

path = sys.argv[1] if len(sys.argv) > 1 else 'Attention is all you need.pdf'
data = open(path, 'rb').read()

# map object number -> (dict bytes, stream bytes or None)
objs = {}
for m in re.finditer(rb'(\d+)\s+(\d+)\s+obj\b', data):
    num = int(m.group(1))
    start = m.end()
    endm = re.search(rb'endobj', data[start:])
    if not endm:
        continue
    body = data[start:start + endm.start()]
    sm = re.search(rb'stream\r?\n', body)
    if sm:
        d = body[:sm.start()]
        raw = body[sm.end():]
        raw = re.sub(rb'\r?\n?endstream\s*$', b'', raw)
        objs[num] = (d, raw)
    else:
        objs[num] = (body, None)

def inflate(raw):
    for attempt in (raw, raw.strip()):
        try:
            return zlib.decompress(attempt)
        except Exception:
            pass
    try:
        return zlib.decompressobj().decompress(raw)
    except Exception:
        return b''

# collect pages in document order from /Type /Page objects
pages = []
for num, (d, raw) in objs.items():
    if re.search(rb'/Type\s*/Page[^s]', d):
        pages.append(num)

def getref(key, d):
    m = re.search(key.encode() + rb'\s+(\d+)\s+0\s+R', d)
    return int(m.group(1)) if m else None

def resolve(num, seen=None):
    seen = seen or set()
    if num is None or num in seen or num not in objs:
        return None
    seen.add(num)
    d, raw = objs[num]
    if raw is not None:
        return inflate(raw)
    # object stream?
    if b'/ObjStm' in d:
        content = inflate(raw) if raw is not None else b''
        return content
    return None

def contents_of(pnum):
    d, raw = objs[pnum]
    m = re.search(rb'/Contents\s+(\d+)\s+0\s+R', d)
    parts = []
    if m:
        parts.append(resolve(int(m.group(1))) or b'')
    else:
        m = re.search(rb'/Contents\s*\[(.*?)\]', d, re.S)
        if m:
            for r in re.finditer(rb'(\d+)\s+0\s+R', m.group(1)):
                parts.append(resolve(int(r.group(1))) or b'')
    # merge indirect refs that point at arrays (rare)
    out = []
    for p in parts:
        if p and not p.strip().startswith(b'['):
            out.append(p)
    return b'\n'.join(out)

def unescape(s):
    out = bytearray()
    i = 0
    while i < len(s):
        c = s[i]
        if c == 0x5c and i + 1 < len(s):
            n = s[i + 1]
            mp = {0x6e: 10, 0x72: 13, 0x74: 9, 0x62: 8, 0x66: 12,
                  0x28: 40, 0x29: 41, 0x5c: 92}
            if n in mp:
                out.append(mp[n]); i += 2; continue
            if 0x30 <= n <= 0x37:
                oct_ = s[i + 1:i + 4]
                k = 0
                while k < 3 and k < len(oct_) and 0x30 <= oct_[k] <= 0x37:
                    k += 1
                out.append(int(oct_[:k], 8) & 0xFF); i += 1 + k; continue
            out.append(n); i += 2; continue
        out.append(c); i += 1
    return bytes(out)

lines = []
for idx, pnum in enumerate(pages):
    content = contents_of(pnum)
    txt = []
    # extract strings inside BT..ET, tracking TJ/Tj
    for tm in re.finditer(rb'\((?:[^()\\]|\\.)*\)|<[0-9A-Fa-f\s]+>|\bTd\b|\bTD\b|\bT\*\b|\bTJ\b|\bTj\b|\bET\b', content, re.S):
        tok = tm.group(0)
        if tok.startswith(b'('):
            txt.append(unescape(tok[1:-1]))
        elif tok.startswith(b'Td') or tok.startswith(b'TD') or tok == b'T*':
            txt.append(b' ')
    s = b''.join(txt)
    s = re.sub(rb'[^\x20-\x7e\n]+', b' ', s)
    lines.append('=== PAGE %d (obj %d) ===\n' % (idx + 1, pnum) + s.decode('latin-1'))

out = '\n'.join(lines)
open('_tools/pdf_text.txt', 'w', encoding='utf-8', errors='replace').write(out)
print('pages:', len(pages), 'chars:', len(out))
