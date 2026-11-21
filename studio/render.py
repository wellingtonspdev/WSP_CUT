"""Turns a time range into a vertical clip: 9:16 layout, bouncy word captions, hook title, cover."""
import re
import shutil
import subprocess
import tempfile
import unicodedata
from pathlib import Path

from . import config, roman, thumbs
from .media import NO_WINDOW, run

W, H = config.OUT_W, config.OUT_H

# ASS colours are &HBBGGRR. `size` is the ASS font size (line height); _Font turns it into real pixels.
STYLES = {
    "bounce": {"label": "Bounce", "font": "Poppins ExtraBold", "size": 132, "base": "FFFFFF", "active": "00F0FF",
               "strong": "FFFFFF", "outline": 7, "pop": 112, "upper": True, "per": 3},
    "hype": {"label": "Hype", "font": "Anton", "size": 152, "base": "FFFFFF", "active": "3C3CFF",
             "strong": "00D7FF", "outline": 7, "pop": 114, "upper": True, "per": 2},
    "clean": {"label": "Clean", "font": "Poppins ExtraBold", "size": 100, "base": "FFFFFF", "active": "6BD7FF",
              "strong": "FFFFFF", "outline": 5, "pop": 106, "upper": False, "per": 4},
}
HOOK = {"font": "Poppins ExtraBold", "size": 96, "ink": "14100E", "box": "FFFFFF", "pad_x": 36, "pad_y": 22, "radius": 26,
        "max_w": 900}
FONT_FILES = {"Poppins ExtraBold": "Poppins-ExtraBold.ttf", "Anton": "Anton-Regular.ttf", "Archivo Black": "ArchivoBlack-Regular.ttf"}
CAP_MAX_W = 960   # captions never run wider than this, whatever is said


class _Font:
    """Measures text exactly as the renderer will draw it, so every word can be placed by hand: the highlighted
    word can grow without pushing its neighbours, and the heading box fits its text."""
    _cache: dict = {}

    def __init__(self, name):
        import struct
        path = Path(__file__).resolve().parent / "fonts" / FONT_FILES.get(name, FONT_FILES["Poppins ExtraBold"])
        d = path.read_bytes()
        tables = {}
        for i in range(struct.unpack(">H", d[4:6])[0]):
            tag, _, off, _ = struct.unpack(">4sIII", d[12 + 16 * i: 28 + 16 * i])
            tables[tag] = off
        self.upm = struct.unpack(">H", d[tables[b"head"] + 18: tables[b"head"] + 20])[0]
        o = tables[b"OS/2"]
        self.asc, self.desc = struct.unpack(">HH", d[o + 74: o + 78])      # what the renderer sizes the font by
        self.cap = struct.unpack(">h", d[o + 88: o + 90])[0] or int(self.upm * 0.7)
        self.path = str(path)
        self._pil = {}

    @classmethod
    def get(cls, name):
        if name not in cls._cache:
            cls._cache[name] = cls(name)
        return cls._cache[name]

    def em(self, size):
        return size * self.upm / (self.asc + self.desc)

    def width(self, text, size):
        """Pixels wide at this ASS size."""
        try:
            from PIL import ImageFont
            if "f" not in self._pil:
                self._pil["f"] = ImageFont.truetype(self.path, 200)
            return self._pil["f"].getlength(text) * self.em(size) / 200
        except Exception:  # noqa: BLE001 - no measuring library: a fair estimate keeps clips rendering
            return len(text) * self.em(size) * 0.62

    def cap_shift(self, size):
        """How far below the anchor the middle of a capital letter sits when text is centred on a point."""
        return ((self.asc - self.desc) / 2 - self.cap / 2) * size / (self.asc + self.desc)


CAPTION_LEAD = 0.0  # measured: word stamps are within 0.15 s of the voice, so no shift is needed
def install_fonts():
    """Copy the bundled caption fonts next to the data, so captions look the same on every machine and server."""
    src = Path(__file__).resolve().parent / "fonts"
    for f in src.glob("*.ttf"):
        if not (config.FONTS_DIR / f.name).exists():
            shutil.copyfile(f, config.FONTS_DIR / f.name)


LAYOUTS = {"blur": "Full frame + blurred fill", "zoom": "Zoomed in + blurred fill", "crop": "Fill the screen (centre crop)",
           "gameplay": "Gameplay fills the screen, facecam in the corner",
           "horizontal": "Horizontal 16:9 (full frame for YouTube)"}


def _ts(t):
    t = max(0.0, t)
    h, m = int(t // 3600), int(t % 3600 // 60)
    return f"{h}:{m:02d}:{t % 60:05.2f}"


def _clean(text):
    return re.sub(r"[\x00-\x1f\x7f]+", " ", re.sub(r"[{}\\]", "", str(text))).strip()


def _key(word):
    folded = "".join(ch for ch in unicodedata.normalize("NFD", str(word or "")) if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9']", "", folded.lower())


def _groups(words, per):
    out, cur = [], []
    for w in words:
        if cur and (len(cur) >= per or sum(len(x["w"]) for x in cur) + len(w["w"]) > 16
                    or w["s"] - cur[-1]["e"] > 0.45 or cur[-1]["w"][-1:] in ".?!"):
            out.append(cur)
            cur = []
        cur.append(w)
    if cur:
        out.append(cur)
    return out


def geometry(src_w, src_h, layout):
    """Where the video sits in the output frame, and where hook and captions go."""
    if layout == "horizontal":
        out_h = 1080 if src_h >= 1080 else 720
        return "horizontal", out_h, 0, out_h - 90
    if src_h >= src_w and layout != "crop":
        layout = "crop"  # already vertical or square: nothing to gain from a blurred fill
    if layout == "blur":
        fg_h = min(H, int(round(W * src_h / src_w / 2)) * 2)
    elif layout == "zoom":
        fg_h = 1000
    else:
        fg_h = H
    top = (H - fg_h) // 2
    if layout in ("crop", "gameplay"):
        return layout, fg_h, 300, 1380
    return layout, fg_h, max(230, top - 190), min(1640, top + fg_h + 180)


def _hook_events(hook, dur, y):
    """The heading: dark text in a white rounded box that fits it, on one line or two balanced ones."""
    f, size = _Font.get(HOOK["font"]), HOOK["size"]
    words = hook.split()
    if not words:
        return []
    lines = [" ".join(words)]
    if f.width(lines[0], size) > HOOK["max_w"] and len(words) > 1:      # two lines, split where they come out most even
        cut = min(range(1, len(words)), key=lambda i: max(f.width(" ".join(words[:i]), size), f.width(" ".join(words[i:]), size)))
        lines = [" ".join(words[:cut]), " ".join(words[cut:])]
    widest = max(f.width(l, size) for l in lines)
    if widest > HOOK["max_w"]:
        size = max(40, int(size * HOOK["max_w"] / widest))
        widest = max(f.width(l, size) for l in lines)
    line_h = f.em(size) * 1.16
    bw, bh = widest + 2 * HOOK["pad_x"], line_h * len(lines) + 2 * HOOK["pad_y"]
    x0, y0, r = (W - bw) / 2, y - bh / 2, min(HOOK["radius"], bh / 2)
    x1, y1 = x0 + bw, y0 + bh
    n = lambda v: f"{v:.0f}"                                                         # noqa: E731
    box = (f"m {n(x0 + r)} {n(y0)} l {n(x1 - r)} {n(y0)} b {n(x1)} {n(y0)} {n(x1)} {n(y0)} {n(x1)} {n(y0 + r)} "
           f"l {n(x1)} {n(y1 - r)} b {n(x1)} {n(y1)} {n(x1)} {n(y1)} {n(x1 - r)} {n(y1)} "
           f"l {n(x0 + r)} {n(y1)} b {n(x0)} {n(y1)} {n(x0)} {n(y1)} {n(x0)} {n(y1 - r)} "
           f"l {n(x0)} {n(y0 + r)} b {n(x0)} {n(y0)} {n(x0)} {n(y0)} {n(x0 + r)} {n(y0)}")
    span = f"{_ts(0)},{_ts(min(dur, 7.0))}"
    ev = [f"Dialogue: 3,{span},Hook,,0,0,0,,{{\\an7\\pos(0,0)\\fad(160,250)\\c&H{HOOK['box']}&\\p1}}{box}{{\\p0}}"]
    for i, line in enumerate(lines):
        cy = y - line_h * (len(lines) - 1) / 2 + i * line_h - f.cap_shift(size)
        ev.append(f"Dialogue: 4,{span},Hook,,0,0,0,,{{\\an5\\pos({W // 2},{cy:.0f})\\fad(160,250)\\fs{size}}}{line}")
    return ev


def build_ass(words, start, end, hook, style_name, highlight, hook_y, cap_y) -> str:
    st = STYLES.get(style_name, STYLES["bounce"])
    words = [dict(w, s=w["s"] - CAPTION_LEAD, e=w["e"] - CAPTION_LEAD) for w in words]
    strong = {_key(h) for h in highlight if _key(h)}
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{st['font']},{st['size']},&H00{st['base']},&H00{st['base']},&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,{st['outline']},2,5,0,0,0,1
Style: Hook,{HOOK['font']},{HOOK['size']},&H00{HOOK['ink']},&H00{HOOK['ink']},&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    dur = end - start
    ev = _hook_events(_clean(hook), dur, hook_y)
    f = _Font.get(st["font"])
    inside = [dict(w, w=_clean(w["w"])) for w in words if w["s"] >= start - 0.05 and w["s"] < end]
    inside = sorted((w for w in inside if w["w"]), key=lambda w: w["s"])
    for i, w in enumerate(inside[:-1]):
        w["e"] = max(w["s"], min(w["e"], inside[i + 1]["s"]))
    groups = _groups(inside, st["per"])
    for gi, g in enumerate(groups):
        g_start = max(0.0, g[0]["s"] - start)
        g_end = min(g[-1]["e"] + 0.25, groups[gi + 1][0]["s"] if gi + 1 < len(groups) else end) - start
        if g_end - g_start < 0.02:
            continue
        texts = []
        for x in g:
            t = re.sub(r"[.,;:]+$", "", x["w"])
            texts.append(t.upper() if st["upper"] else t)
        if not all(roman.drawable(t) for t in texts):
            # A script the caption font cannot draw (Japanese, Thai, Chinese...): its words cannot be measured, so
            # they are not placed one by one. The line is drawn whole and centred, in a font the system has for it.
            line = " ".join(texts)
            fs = max(44, min(int(st["size"] * 0.7), int(1500 / max(1, len(line)) * 1.6)))
            ev.append(f"Dialogue: 1,{_ts(g_start)},{_ts(g_end)},Cap,,60,60,0,,{{\\an5\\pos({W // 2},{cap_y:.0f})\\q0\\fs{fs}}}{line}")
            continue
        size = st["size"]
        gap = f.width("n n", size) - 2 * f.width("n", size) + (2 * st["outline"] + 6) * size / st["size"]   # a space, plus room for both outlines
        total = sum(f.width(t, size) for t in texts) + gap * (len(texts) - 1)
        if total > CAP_MAX_W:                                  # a long phrase: shrink it to fit, never run off the frame
            size = max(36, int(size * CAP_MAX_W / total))
            gap = f.width("n n", size) - 2 * f.width("n", size) + (2 * st["outline"] + 6) * size / st["size"]   # a space, plus room for both outlines
            total = sum(f.width(t, size) for t in texts) + gap * (len(texts) - 1)
        cy = cap_y - f.cap_shift(size)
        x = (W - total) / 2
        p = st["pop"]
        for i, (w, t) in enumerate(zip(g, texts)):
            wd = f.width(t, size)
            p = max(104, min(st["pop"], 100 + int(2200 / max(wd, 1))))   # long words grow less, so they never touch their neighbours
            pos = f"\\an5\\pos({x + wd / 2:.0f},{cy:.0f})\\fs{size}"
            x += wd + gap
            a = max(g_start, w["s"] - start)
            b = (g[i + 1]["s"] - start) if i + 1 < len(g) else g_end
            base = st["strong"] if _key(w["w"]) in strong else st["base"]
            if a - g_start >= 0.02:                                                   # waiting to be said
                ev.append(f"Dialogue: 1,{_ts(g_start)},{_ts(a)},Cap,,0,0,0,,{{{pos}\\c&H{base}&}}{t}")
            if b - a >= 0.02:                                                         # being said: colour and a quick pop, in front
                ev.append(f"Dialogue: 2,{_ts(a)},{_ts(b)},Cap,,0,0,0,,{{{pos}\\c&H{st['active']}&\\fscx94\\fscy94"
                          f"\\t(0,90,\\fscx{p}\\fscy{p})\\t(90,200,\\fscx{p - 5}\\fscy{p - 5})}}{t}")
            if g_end - b >= 0.02:                                                     # said
                ev.append(f"Dialogue: 1,{_ts(b)},{_ts(g_end)},Cap,,0,0,0,,{{{pos}\\c&H{base}&\\fscx{p - 5}\\fscy{p - 5}"
                          f"\\t(0,90,\\fscx100\\fscy100)}}{t}")
    return head + "\n".join(ev) + "\n"


PIP_W = 380            # how wide the facecam is in the 1080-wide frame
PIP_MARGIN = 26


def _filter(layout, fg_h, facecam=None):
    if layout == "horizontal":
        out_w, out_h = (1920, 1080) if fg_h >= 1080 else (1280, 720)
        return (f"[0:v]fps={config.OUT_FPS},scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,"
                f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2,setsar=1[v0];")
    if layout == "gameplay" and facecam:
        # The game fills the screen, and the webcam sits in the top corner of it, so a viewer sees
        # the play and the face at the same time.
        fx, fy, fw, fh = (int(v) for v in facecam)
        if fw > 8 and fh > 8:
            pip_h = max(2, int(round(PIP_W * fh / fw)))
            return (f"[0:v]fps={config.OUT_FPS},split=2[base][fc];"
                    f"[base]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1[b];"
                    f"[fc]crop={fw}:{fh}:{fx}:{fy},scale={PIP_W}:{pip_h},setsar=1[p];"
                    f"[b][p]overlay={W - PIP_W - PIP_MARGIN}:{PIP_MARGIN}[v0];")
    if layout == "crop" or (layout == "gameplay" and not facecam):
        return f"[0:v]fps={config.OUT_FPS},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1[v0];"
    return (f"[0:v]fps={config.OUT_FPS},split=2[a][b];"
            f"[a]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=6:2,"
            f"scale={W}:{H},eq=brightness=-0.14:saturation=1.15[bg];"
            f"[b]scale={W}:{fg_h}:force_original_aspect_ratio=increase,crop={W}:{fg_h}[fg];"
            f"[bg][fg]overlay=0:(H-h)/2,setsar=1[v0];")


def _hook_y(layout, hook_y, facecam=None):
    """Where the heading goes. In the gameplay layout the webcam sits in the top corner, so a heading
    across the top would run under it: it goes just below the picture instead."""
    if layout == "gameplay" and facecam and len(facecam) == 4 and facecam[2] > 8:
        return PIP_MARGIN + int(round(PIP_W * facecam[3] / facecam[2])) + 32
    return hook_y


_qsv_ok = None
RENDER_SECONDS = 600      # one clip's video encode; far beyond normal, so a frozen encoder is stopped, not waited on


def _encode_cmd(work: Path, base: list, out_file: str, enc_timeout: int):
    global _qsv_ok
    encoders = []
    if _qsv_ok is not False:
        encoders.append(("qsv", ["nv12", "-c:v", "h264_qsv", "-global_quality", "23", "-preset", "fast"]))
    encoders.append(("x264", ["yuv420p", "-c:v", "libx264", "-crf", "21", "-preset", "veryfast"]))
    err = None
    for label, enc in encoders:
        p = subprocess.run(base + enc + [out_file], cwd=work, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", creationflags=NO_WINDOW, timeout=enc_timeout)
        if p.returncode == 0:
            if label == "qsv":
                _qsv_ok = True
            return
        err = p.stderr[-1200:]
        if label == "qsv":
            _qsv_ok = False
    raise RuntimeError(f"Render failed: {err}")


def _scaled_facecam(face, src_w: int, src_h: int, out_w: int = 1280, out_h: int = 720):
    if not face or len(face) != 4 or not src_w or not src_h:
        return None
    sx, sy = float(out_w) / src_w, float(out_h) / src_h
    return [round(face[0] * sx), round(face[1] * sy), round(face[2] * sx), round(face[3] * sy)]


def _thumb_for_long(out_dir: Path, name: str, clip: dict, peak_t: float, src_w: int, src_h: int,
                    out_w: int = 1280, out_h: int = 720):
    out_mp4 = out_dir / f"{name}.mp4"
    thumbs.generate(
        out_mp4,
        out_dir / f"{name}.jpg",
        peak_t=peak_t,
        text=clip.get("thumb_text") or clip.get("hook") or clip.get("title") or "",
        facecam=_scaled_facecam(clip.get("facecam"), src_w, src_h, out_w, out_h),
        mode=clip.get("thumb_mode") or "local",
    )


def render_compilation(src: Path, out_dir: Path, name: str, clip: dict, src_w: int, src_h: int,
                       default_cover: Path | None = None) -> dict:
    """Concatenate multiple highlight segments into a single 1920x1080 16:9 MP4 normalized to -14 LUFS."""
    segments = clip.get("segments") or []
    if not segments:
        raise ValueError("No segments provided for compilation.")
    out_dir.mkdir(parents=True, exist_ok=True)
    tbase = config.TMP_DIR if len(str(config.TMP_DIR)) < 120 else None
    if tbase:
        tbase.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="pk", dir=str(tbase) if tbase else None))
    try:
        fc = _filter("horizontal", 1080) + "[v0]null[v]"
        part_files = []
        total_dur = 0.0
        cover_t = None
        peak_t = clip.get("peak_t")
        for idx, seg in enumerate(segments):
            seg_src = Path(seg["src"]) if seg.get("src") and Path(seg["src"]).is_file() else src
            seg_off = float(seg.get("src_offset", 0.0)) if seg_src != src else 0.0
            s = max(0.0, float(seg["start"]) - seg_off)
            e = max(s + 0.1, float(seg["end"]) - seg_off)
            dur = max(0.1, e - s)
            if cover_t is None and peak_t is not None and float(seg["start"]) <= float(peak_t) <= float(seg["end"]):
                cover_t = max(0.0, total_dur + (float(peak_t) - float(seg["start"])))
            total_dur += dur
            part_name = f"seg_{idx:03d}.mp4"
            enc_timeout = max(RENDER_SECONDS, int(dur * 2.5))
            base = ["ffmpeg", "-y", "-v", "error", "-ss", f"{s:.2f}", "-t", f"{dur:.2f}", "-i", str(seg_src),
                    "-filter_complex", fc, "-map", "[v]", "-map", "0:a:0", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
                    "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2", "-pix_fmt"]
            _encode_cmd(work, base, part_name, enc_timeout)
            part_files.append(part_name)
        lst = work / "concat.txt"
        lst.write_text("".join(f"file '{p}'\n" for p in part_files), encoding="utf-8")
        run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", "concat.txt",
             "-c", "copy", "-movflags", "+faststart", f"{name}.mp4"], cwd=work)
        if cover_t is None:
            cover_t = min(1.2, total_dur / 2)
        cover_t = max(0.0, min(max(0.0, total_dur - 0.1), cover_t))
        run(["ffmpeg", "-y", "-v", "error", "-ss", f"{cover_t:.2f}", "-i", f"{name}.mp4", "-frames:v", "1",
             "-q:v", "3", f"{name}.jpg"], cwd=work)
        for ext in ("mp4", "jpg"):
            f = work / f"{name}.{ext}"
            if f.exists():
                shutil.move(str(f), str(out_dir / f"{name}.{ext}"))
        if default_cover and Path(default_cover).is_file():
            shutil.copyfile(str(default_cover), str(out_dir / f"{name}.jpg"))
        else:
            _thumb_for_long(out_dir, name, clip, cover_t, src_w, src_h, out_w=1920, out_h=1080)
        return {"file": f"{name}.mp4", "cover": f"{name}.jpg", "layout": "horizontal"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def render_clip(src: Path, out_dir: Path, name: str, clip: dict, words: list, src_w: int, src_h: int,
                aspect: str | None = None, default_cover: Path | None = None, cover_first_frame: bool = False) -> dict:
    """Writes <name>.mp4, <name>.jpg and <name>.ass into out_dir. Returns file names."""
    if clip.get("segments"):
        return render_compilation(src, out_dir, name, clip, src_w, src_h, default_cover=default_cover)
    out_dir.mkdir(parents=True, exist_ok=True)
    start, end = clip["start"], clip["end"]
    is_16_9 = aspect == "16:9" or clip.get("format") == "16:9" or clip.get("layout") == "horizontal"
    req_layout = "horizontal" if is_16_9 else clip.get("layout", "blur")
    layout, fg_h, hook_y, cap_y = geometry(src_w, src_h, req_layout)
    if is_16_9:
        fg_h = 1080
    face = clip.get("facecam")
    if layout == "gameplay" and not (face and len(face) == 4):
        layout, fg_h, hook_y, cap_y = geometry(src_w, src_h, "blur")  # no single webcam found (e.g. multi-cam or HUD): keep full 16:9 frame
    hook_y = _hook_y(layout, hook_y, face)
    if layout == "horizontal":
        ass = build_ass(words if clip.get("captions", False) else [], start, end, "", clip.get("style", "clean"),
                        clip.get("highlight_words", []), hook_y, cap_y) if clip.get("captions", False) else ""
    else:
        ass = build_ass(words if clip.get("captions", True) else [], start, end, clip.get("hook", ""), clip.get("style", "bounce"),
                        clip.get("highlight_words", []), hook_y, cap_y)
    # The clip is made in a short work folder and moved into place afterwards. The caption step of ffmpeg cannot
    # open files whose full path is longer than 260 characters on Windows, which a data folder deep inside other
    # folders easily reaches. Inside the work folder every name is short and relative (a drive colon would also
    # break the filter syntax). Prefer PeakCut's own short TMP_DIR when short enough so a full C: drive never stops rendering.
    tbase = config.TMP_DIR if len(str(config.TMP_DIR)) < 120 else None
    if tbase:
        tbase.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="pk", dir=str(tbase) if tbase else None))
    try:
        res = _render_in(work, src, out_dir, name, start, end, layout, fg_h, ass, face=face, peak_t=clip.get("peak_t"),
                         default_cover=default_cover, cover_first_frame=bool(cover_first_frame and not is_16_9))
        if is_16_9 or clip.get("clip_kind") == "long":
            if default_cover and Path(default_cover).is_file():
                shutil.copyfile(str(default_cover), str(out_dir / f"{name}.jpg"))
            else:
                pt = clip.get("peak_t")
                if pt is not None and start <= float(pt) <= end:
                    local_peak_t = max(0.0, min(max(0.0, (end - start) - 0.1), float(pt) - start))
                else:
                    local_peak_t = min(max(3.0, (end - start) * 0.35), (end - start) / 2)
                out_w, out_h = (1920, 1080) if fg_h >= 1080 else (1280, 720)
                _thumb_for_long(out_dir, name, clip, local_peak_t, src_w, src_h, out_w=out_w, out_h=out_h)
        return res
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _render_in(work: Path, src: Path, out_dir: Path, name: str, start, end, layout, fg_h, ass: str,
               face=None, peak_t=None, default_cover: Path | None = None, cover_first_frame: bool = False) -> dict:
    if ass:
        (work / f"{name}.ass").write_text(ass, encoding="utf-8")
        (work / "f").mkdir()
        for f in config.FONTS_DIR.glob("*.ttf"):
            shutil.copyfile(f, work / "f" / f.name)
        fc = _filter(layout, fg_h, face) + f"[v0]ass={name}.ass:fontsdir=f[vm]"
    else:
        fc = _filter(layout, fg_h, face) + "[v0]null[vm]"
    enc_timeout = max(RENDER_SECONDS, int((end - start) * 2.5))
    has_dc = bool(default_cover and Path(default_cover).is_file())
    if has_dc and cover_first_frame:
        shutil.copyfile(str(default_cover), str(work / "cover_ff.jpg"))
        fc_full = (
            fc + ";"
            f"[1:v]fps={config.OUT_FPS},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1[vc];"
            "[2:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[ac];"
            "[0:a:0]loudnorm=I=-14:TP=-1.5:LRA=11,aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[am];"
            "[vc][ac][vm][am]concat=n=2:v=1:a=1[v][a]"
        )
        base = ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.2f}", "-t", f"{end - start:.2f}", "-i", str(src),
                "-loop", "1", "-framerate", str(config.OUT_FPS), "-t", "0.12", "-i", "cover_ff.jpg",
                "-f", "lavfi", "-t", "0.12", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
                "-filter_complex", fc_full, "-map", "[v]", "-map", "[a]",
                "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-movflags", "+faststart", "-pix_fmt"]
    else:
        base = ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.2f}", "-t", f"{end - start:.2f}", "-i", str(src),
                "-filter_complex", fc, "-map", "[vm]", "-map", "0:a:0", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
                "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-movflags", "+faststart", "-pix_fmt"]
    _encode_cmd(work, base, f"{name}.mp4", enc_timeout)
    if has_dc:
        shutil.copyfile(str(default_cover), str(work / f"{name}.jpg"))
    else:
        if peak_t is not None and start <= float(peak_t) <= end:
            cover_t = max(0.0, min(max(0.0, (end - start) - 0.1), float(peak_t) - start))
        else:
            cover_t = min(max(3.0, (end - start) * 0.35), (end - start) / 2) if layout == "horizontal" else min(1.2, (end - start) / 2)
        run(["ffmpeg", "-y", "-v", "error", "-ss", f"{cover_t:.2f}", "-i", f"{name}.mp4", "-frames:v", "1",
             "-q:v", "3", f"{name}.jpg"], cwd=work)
    for ext in ("mp4", "jpg", "ass"):
        f = work / f"{name}.{ext}"
        if f.exists():
            shutil.move(str(f), str(out_dir / f"{name}.{ext}"))
    return {"file": f"{name}.mp4", "cover": f"{name}.jpg", "layout": layout, "has_default_cover": has_dc}
