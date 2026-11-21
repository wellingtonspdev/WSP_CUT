"""Job pipeline: get video -> audio -> transcript -> AI picks -> rendered clips. One heavy task at a time."""
import collections
import json
import os
import math
import queue
import re
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import brain, config, creators, ctx, facecam, games, load, media, quality, render, roman, safe, thumbs, transcribe, usage

class _FairQueue:
    """Whose video is next. Each person has their own line and the lines take turns, so someone who adds ten
    videos does not make everyone else wait behind all ten. Small tasks (re-making one clip) go first."""

    def __init__(self):
        self.cv = threading.Condition()
        self.quick = collections.deque()
        self.lines = {}                       # user -> their waiting videos, oldest first
        self.turn = collections.deque()       # users with something waiting, in the order they will be served
        self.epoch = 0                        # goes up whenever the line moves
        self._pos = (0.0, {})
        self.running = 0                      # videos being worked on right now
        self.last_start = 0.0
        self.held = False                     # videos are waiting only because the machine is near full

    def _room(self) -> bool:
        """May one more video start? One always may. More only while the machine has room, and a few seconds
        after the last one started, so its weight is already showing in the measurements."""
        if self.running == 0:
            return True
        return self.running < config.JOB_WORKERS and time.time() - self.last_start > 6 and not load.busy()

    def done(self):
        with self.cv:
            self.running = max(0, self.running - 1)
            self.cv.notify_all()

    def put(self, item, user=None):
        with self.cv:
            if user is None:
                self.quick.append(item)
            else:
                if user not in self.lines:
                    self.lines[user] = collections.deque()
                    self.turn.append(user)
                self.lines[user].append(item)
            self.epoch += 1
            self.cv.notify()

    def get(self, quick_only=False):
        with self.cv:
            while True:
                if self.quick:
                    self.epoch += 1
                    return self.quick.popleft()
                if not quick_only and self.turn:
                    if self._room():
                        break
                    if not self.held:
                        self.held = True
                        self.epoch += 1
                elif self.held:
                    self.held = False
                    self.epoch += 1
                self.cv.wait(1.0)             # looked at again every second: the line moves the moment there is room
            self.epoch += 1
            self.held = False
            self.running += 1
            self.last_start = time.time()
            user = self.turn.popleft()
            item = self.lines[user].popleft()
            if self.lines[user]:
                self.turn.append(user)
            else:
                del self.lines[user]
            return item

    def ahead(self) -> dict:
        """job id -> how many videos will start before it. Worked out at most once a second."""
        with self.cv:
            if time.time() - self._pos[0] < 1:
                return self._pos[1]
            out, lines, order, n = {}, {u: list(q) for u, q in self.lines.items()}, list(self.turn), 0
            depth = 0
            while order:
                nxt = []
                for u in order:
                    if depth < len(lines[u]):
                        out[lines[u][depth][1]] = n
                        n += 1
                        if depth + 1 < len(lines[u]):
                            nxt.append(u)
                order, depth = nxt, depth + 1
            self._pos = (time.time(), out)
            return out


_q = _FairQueue()
_live_q: "queue.Queue" = queue.Queue()       # live short clips never wait behind a long video
_live_long_q: "queue.Queue" = queue.Queue()  # long 10-20 min live renders run on their own worker
_jobs: dict = {}
_by_user: dict = {}                       # user -> {job id: job}: listing one person's videos never scans everyone's
_lock = threading.RLock()
on_clip_ready: list = []   # callbacks (job, clip), run once when a clip first finishes

DEFAULTS = {"max_clips": 6, "min_len": 15, "max_len": 60, "style": "bounce", "layout": "blur", "quality": 720,
            "range_start": None, "range_end": None, "captions": True, "output_lang": config.OUTPUT_LANG,
            "long_mode": "off", "long_min_len": 600, "long_max_len": 1200, "derive_shorts": True, "thumb_mode": "local"}


LIMITS = {"max_clips": (1, 20), "min_len": (5, 600), "max_len": (10, 900), "quality": (360, 1080),
          "long_min_len": (300, 1200), "long_max_len": (600, 1800)}


def clean_options(options) -> dict:
    """Only known settings, each inside its allowed range. Raises ValueError on nonsense."""
    out = {}
    for k, v in (options if isinstance(options, dict) else {}).items():
        if k in ("captions", "derive_shorts"):
            if not isinstance(v, bool):
                label = "Captions" if k == "captions" else "Derive shorts"
                raise ValueError(f"{label} must be on or off.")
            out[k] = v
            continue
        if k not in DEFAULTS or v in ("", None):
            continue
        if k in LIMITS:
            lo, hi = LIMITS[k]
            v = max(lo, min(hi, int(float(v))))
        elif k == "style":
            v = v if v in render.STYLES else DEFAULTS["style"]
        elif k == "layout":
            v = v if v in render.LAYOUTS else DEFAULTS["layout"]
        elif k == "long_mode":
            v = v if v in ("off", "block", "compile", "both") else DEFAULTS["long_mode"]
        elif k == "thumb_mode":
            v = v if v in ("local", "ai") else DEFAULTS["thumb_mode"]
        elif k == "output_lang":
            v = brain.clean_output_lang(v)
        else:                                    # range_start, range_end: seconds into the video
            v = float(v)
            if not math.isfinite(v) or v < 0:
                raise ValueError("That part of the video isn't valid.")
        out[k] = v
    if out.get("range_end") is not None and out["range_end"] <= out.get("range_start", 0):
        raise ValueError("The end of the chosen part must be after its start.")
    return out


def job_dir(jid) -> Path:
    return Path(_jobs[jid]["dir"])


HIDDEN = ("dir", "user", "source_path", "recover", "pending_recut")


def public(job) -> dict:
    """What the browser may see: no server paths, no owner id."""
    out = {k: v for k, v in job.items() if k not in HIDDEN}
    out["source"] = {k: v for k, v in job["source"].items() if k != "path"}
    jdir = Path(job["dir"]) if job.get("dir") else None
    has_job_src = bool(job.get("source_path") and Path(job["source_path"]).is_file())
    has_url_src = bool((job.get("source") or {}).get("type") == "url" and (job.get("source") or {}).get("url"))
    out["has_cache"] = bool(jdir and (jdir / "words.json").is_file() and (has_job_src or has_url_src))
    out["clips"] = [
        {
            **{k: v for k, v in c.items() if k not in ("src", "fixed_words", "fixed_for")},
            "local": bool(
                jdir
                and c.get("file")
                and Path(str(c["file"])).name
                and (jdir / "clips" / Path(str(c["file"])).name).is_file()
            ),
        }
        for c in job["clips"]
    ]
    ai = job.get("ai") or {}
    out["ai"] = {"tokens": ai.get("tokens", 0), "note": ai.get("note", ""), "used": bool(ai.get("model"))}   # never the model, never raw errors
    if job.get("status") == "queued":
        n = _q.ahead().get(job["id"])
        if n:
            out["stage"] = (f"High demand right now: {n} video{'s' if n != 1 else ''} ahead of yours" if _q.held else
                            f"Waiting in line: {n} video{'s' if n != 1 else ''} ahead of yours")
        elif n == 0 and _q.held:
            out["stage"] = "High demand right now: yours is next"
    return out


def save(job):
    ctx.bump(job.get("user"))
    with _lock:
        d = Path(job["dir"])
        for _ in range(5):
            try:
                text = json.dumps(job, indent=1)
                break
            except RuntimeError:  # changed mid-dump by another thread
                time.sleep(0.05)
        else:
            return
        tmp = d / "job.json.tmp"
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(d / "job.json")


def _set(job, **kw):
    with _lock:
        job.update(kw)
    save(job)


def _remap(path, folder: Path):
    """Paths saved before data moved into per-user folders (or on another machine): find the file by name."""
    if path and not Path(path).exists():
        cand = folder / Path(str(path).replace("\\", "/")).name
        if cand.exists():
            return str(cand)
    return path


def _load_file(f: Path, uid: str):
    try:
        job = json.loads(f.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return
    job.update(dir=str(f.parent), user=uid)
    job.setdefault("kind", "video")
    job.setdefault("creator", None)
    job.setdefault("options", {}).setdefault("captions", True)
    job["options"].setdefault("output_lang", config.OUTPUT_LANG)
    if job.get("source_path"):
        job["source_path"] = _remap(job["source_path"], f.parent)
    if job.get("source", {}).get("path"):
        job["source"]["path"] = _remap(job["source"]["path"], f.parent)
    if job.get("status") in ("queued", "running") and job.get("kind") != "live":
        if job.get("pending_recut"):
            job.pop("pending_recut", None)
            for c in job.get("clips", []):
                if c.get("status") in ("rendering", "queued"):
                    c.update(status="error", error="Interrupted by a restart. Open it and press Save and re-make.")
            job.update(status="done", stage="Done", error=None)
        else:
            # Picked up again as soon as the server is back (jobs.start), from the last finished part of the transcript.
            job.update(status="queued", stage="Picking up where it stopped", error=None, resume=True)
    if job.get("status") == "live":
        job.update(status="done", stage="Recording paused by a server update", was_live=True)
    buffered = job.get("kind") == "live" and any((f.parent / "buffer").glob("*.ts"))
    for c in job.get("clips", []):
        c.setdefault("captions", job["options"]["captions"])
        c.setdefault("output_lang", job["options"]["output_lang"])
        if c.get("src"):
            c["src"] = _remap(c["src"], f.parent / "clips")
        if c.get("status") in ("rendering", "queued"):
            if buffered and not c.get("src"):
                c["status"] = "queued"  # the stream is still on disk: live.recover() finishes these
                job["recover"] = True
            elif job.get("resume"):
                c["status"] = "queued"  # the video is being picked up again: this clip is made then
            elif job.get("kind") != "live" and (f.parent / "words.json").is_file():
                c["status"] = "queued"  # a single clip being re-made: jobs.start() makes it again
                job.setdefault("remake", []).append(c["id"])
            else:
                c.update(status="error", error="Interrupted by a restart. Open it and press Save and re-make.")
    _jobs[job["id"]] = job
    _by_user.setdefault(uid, {})[job["id"]] = job


def load_user(uid: str):
    d = config.USERS_DIR / uid
    for f in list((d / "jobs").glob("*/job.json")) + list((d / "creators").glob("*/*/*/job.json")):
        _load_file(f, uid)


def load_all():
    for d in config.USERS_DIR.iterdir():
        if d.is_dir():
            load_user(d.name)


def all_jobs() -> list:
    """Every user's jobs. Only for background work; never send this to a browser."""
    with _lock:
        return list(_jobs.values())


def list_jobs():
    u = ctx.uid()
    with _lock:
        return sorted((dict(j) for j in _by_user.get(u, {}).values()), key=lambda j: -j["created"])


def get(jid):
    """The job, only if it belongs to whoever this code is running for."""
    job = _jobs.get(jid)
    return job if job and job["user"] == ctx.uid() else None


def busy_count() -> int:
    u = ctx.uid()
    return sum(1 for j in list(_by_user.get(u, {}).values()) if j["kind"] == "video" and j["status"] in ("queued", "running"))


def waiting_total() -> int:
    return sum(1 for j in list(_jobs.values()) if j["kind"] == "video" and j["status"] == "queued")


def queue_epoch() -> int:
    return _q.epoch


def running_now() -> int:
    return _q.running


def create(source: dict, options: dict, enqueue=True, creator=None, kind="video") -> dict:
    jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:10]
    c = creators.get(creator) if creator else None
    if c:
        d = creators.root(c["id"]) / ("live" if kind == "live" else "videos") / jid
        base = {**DEFAULTS, **{k: c.get(k, DEFAULTS[k]) for k in (
            "max_clips", "min_len", "max_len", "style", "layout", "captions", "output_lang",
            "long_mode", "long_min_len", "long_max_len", "derive_shorts", "thumb_mode")}}
    else:
        d, base, creator = config.user_dir(ctx.uid()) / "jobs" / jid, dict(DEFAULTS), None
    opts = {**base, **clean_options(options)}
    if opts["min_len"] >= opts["max_len"]:
        opts["min_len"] = max(5, opts["max_len"] - 5)
    if opts["long_min_len"] >= opts["long_max_len"]:
        opts["long_min_len"] = max(300, opts["long_max_len"] - 60)
    job = {"id": jid, "user": ctx.uid(), "kind": kind, "creator": creator, "dir": str(d), "created": time.time(), "source": source,
           "options": opts, "status": "queued", "stage": "Waiting", "progress": 0.0,
           "title": source.get("name", ""), "uploader": "", "duration": 0, "clips": [], "error": None, "ai": {}}
    d.mkdir(parents=True)
    with _lock:
        _jobs[jid] = job
        _by_user.setdefault(job["user"], {})[jid] = job
    save(job)
    if enqueue:
        _q.put(("job", jid, None), user=job["user"])
    return job


def enqueue(job):
    save(job)
    _q.put(("job", job["id"], None), user=job["user"])


def submit(fn):
    """Run any heavy task (a render, a cleanup) on the single worker, in order."""
    _q.put(("fn", ctx.bind(fn), None))


def submit_live(fn):
    """Live-stream short renders and cleanups: their own worker, so a long video can never hold them up."""
    _live_q.put(ctx.bind(fn))


def submit_live_long(fn):
    """Long 10-20 min live renders: separate worker so short clips render immediately."""
    _live_long_q.put(ctx.bind(fn))


def _live_worker():
    while True:
        fn = _live_q.get()
        try:
            fn()
        except Exception:  # noqa: BLE001
            traceback.print_exc()


def _live_long_worker():
    while True:
        fn = _live_long_q.get()
        try:
            fn()
        except Exception:  # noqa: BLE001
            traceback.print_exc()


def delete(jid):
    with _lock:
        job = _jobs.pop(jid, None)
        if job:
            _by_user.get(job["user"], {}).pop(jid, None)
    if job:
        ctx.bump(job["user"])
        shutil.rmtree(job["dir"], ignore_errors=True)
    return bool(job)


def request_render(jid, cid, changes: dict):
    with _lock:
        job = _jobs[jid]
        clip = next(c for c in job["clips"] if c["id"] == cid)
        new = dict(clip)
        if "captions" in changes:
            new.update(clean_options({"captions": changes["captions"]}))
        if "output_lang" in changes:
            new["output_lang"] = brain.clean_output_lang(changes["output_lang"])
        for k in ("start", "end"):
            if k in changes:
                new[k] = round(float(changes[k]), 2)
        if "src_offset" in clip:  # live clip: can only move inside the saved piece of stream
            new["start"] = max(new["start"], clip["src_offset"])
            new["end"] = min(new["end"], clip["src_end"])
        if not (math.isfinite(new["start"]) and math.isfinite(new["end"])) or new["start"] < 0:
            raise ValueError("Those times aren't valid.")
        max_dur = 1800 if new.get("format") == "16:9" else (
            3600 if (clip.get("clip_kind") == "long" or changes.get("layout") == "horizontal" or clip.get("layout") == "horizontal") else 180
        )
        if not 3 <= new["end"] - new["start"] <= max_dur:
            raise ValueError(f"A clip can be 3 to {max_dur} seconds long.")
        for k in ("hook", "title", "description", "style", "layout", "thumb_text", "thumb_mode"):
            if k in changes:
                new[k] = str(changes[k])[:{"hook": 60, "title": 100, "description": 2000, "thumb_text": 40, "thumb_mode": 10}.get(k, 20)]
        if new.get("style") not in render.STYLES:
            new["style"] = DEFAULTS["style"]
        if new.get("layout") not in render.LAYOUTS:
            new["layout"] = DEFAULTS["layout"]
        if "layout" in changes:
            new["format"] = "16:9" if new["layout"] == "horizontal" else "9:16"
        if new.get("thumb_mode") not in ("local", "ai"):
            new["thumb_mode"] = job.get("options", {}).get("thumb_mode", DEFAULTS["thumb_mode"])
        has_clip_src = bool(clip.get("src") and Path(clip["src"]).is_file())
        has_job_src = bool(job.get("source_path") and Path(job["source_path"]).is_file())
        has_url_src = (
            job.get("kind") != "live"
            and (job.get("source") or {}).get("type") == "url"
            and bool((job.get("source") or {}).get("url"))
        )
        was_cleaned = (
            bool(clip.get("drive"))
            or any(c.get("drive") for c in job.get("clips", []))
            or (job.get("source_path") == "" and "source_path" in job)
        )
        if was_cleaned and not (has_clip_src or has_job_src or has_url_src):
            raise ValueError(
                "This clip was archived to Google Drive and its raw video was removed to save disk space, so it cannot be re-made."
            )
        clip.update(new, status="queued")
        save(job)
    _q.put(("clip", jid, cid))


def _next_id(job, prefix: str) -> str:
    n = 0
    for c in job.get("clips", []):
        cid = str(c.get("id") or "")
        if cid.startswith(prefix) and cid[len(prefix):].isdigit():
            n = max(n, int(cid[len(prefix):]))
    return f"{prefix}{n + 1:02d}"


def _norm_url(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    u = urlparse(raw)
    host = (u.hostname or "").lower()
    if host in ("youtube.com", "youtu.be") or host.endswith((".youtube.com", ".youtu.be")):
        if host.endswith("youtu.be"):
            vid = u.path.strip("/").split("/")[0]
            if vid:
                return f"yt:{vid}"
        qs = parse_qs(u.query)
        if qs.get("v"):
            return f"yt:{qs['v'][0]}"
        m = re.search(r"/(?:live|shorts|embed|v)/([\w-]{6,})", u.path)
        if m:
            return f"yt:{m.group(1)}"
    return f"{host}{u.path.rstrip('/')}"


def _find_cached_job(job, lo_s: float, hi_s: float) -> dict | None:
    uid = job.get("user")
    target = _norm_url((job.get("source") or {}).get("url") or "")
    if not uid or not target:
        return None
    with _lock:
        cands = list((_by_user.get(uid) or {}).values())
    for other in sorted(cands, key=lambda x: -float(x.get("created") or 0)):
        if other.get("id") == job.get("id") or other.get("kind") == "live":
            continue
        if _norm_url((other.get("source") or {}).get("url") or "") != target:
            continue
        wf = Path(other["dir"]) / "words.json"
        if not wf.is_file():
            continue
        try:
            words = json.loads(wf.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if words and _covers(words, lo_s, hi_s):
            return other
    return None


def _reuse_cached_src(job, clip) -> bool:
    """Reuse an already downloaded .src.mp4 segment from another clip in the same job when it covers [start, end]."""
    cs, ce = float(clip["start"]), float(clip["end"])
    for prev in job.get("clips", []):
        if prev is clip or not prev.get("src"):
            continue
        p = Path(prev["src"])
        if not p.is_file():
            continue
        off = prev.get("src_offset")
        end = prev.get("src_end")
        if off is None or end is None:
            continue
        if float(off) <= cs + 0.05 and float(end) >= ce - 0.05:
            clip.update(src=str(p), src_offset=float(off), src_end=float(end))
            clip["start"] = max(cs, float(off))
            clip["end"] = min(ce, float(end))
            return True
    return False


def render_one(job, clip, words):
    d = Path(job["dir"])
    clip["status"] = "rendering"
    save(job)
    try:
        segs = clip.get("segments") or []
        segs_ready = bool(segs) and all(s.get("src") and Path(s["src"]).is_file() for s in segs)
        if segs_ready:
            src = Path(segs[0]["src"])
            off = 0.0
            local = dict(clip, hook=roman.to_roman(clip.get("hook") or ""))
        else:
            has_clip_src = bool(clip.get("src") and Path(clip["src"]).is_file())
            has_job_src = bool(job.get("source_path") and Path(job["source_path"]).is_file())
            if not has_clip_src and not has_job_src:
                has_clip_src = _reuse_cached_src(job, clip)
            if not has_clip_src and not has_job_src and (
                (not clip.get("src") and not job.get("source_path"))
                or (job.get("source") or {}).get("type") == "url"
            ):
                _piece_again(job, clip)
                has_clip_src = bool(clip.get("src") and Path(clip["src"]).is_file())
            use_clip_src = has_clip_src or bool(clip.get("src") and not has_job_src)
            off = clip.get("src_offset", 0.0) if use_clip_src else 0.0
            src = Path(clip["src"]) if use_clip_src else Path(job["source_path"])
            if not use_clip_src:
                clip["end"] = min(clip["end"], job["duration"])
            local = dict(clip, start=clip["start"] - off, end=clip["end"] - off, hook=roman.to_roman(clip.get("hook") or ""))
            if clip.get("segments") and off:
                local["segments"] = [{"start": round(float(s["start"]) - off, 2), "end": round(float(s["end"]) - off, 2)} for s in clip["segments"]]
        is_16_9 = clip.get("format") == "16:9" or clip.get("layout") == "horizontal"
        need_caps = False if is_16_9 else clip.get("captions", True)
        near = _caption_words(job, clip, words, off) if need_caps else []
        from . import covers
        dc_path = covers.cover_file(job.get("creator"), wide=is_16_9)
        dc_first = False if is_16_9 else bool(covers.should_inject_first_frame(job.get("creator")))
        rc_kw = {"default_cover": dc_path, "cover_first_frame": dc_first} if dc_path else {}
        out = render.render_clip(src, d / "clips", clip["id"], local, near, job.get("width") or 1280, job.get("height") or 720, **rc_kw)
        now_ts = time.time()
        clip.update(out, status="done", version=clip.get("version", 0) + 1, error=None, rendered_at=now_ts)
        if clip["version"] > 1:
            clip.pop("drive", None)
        post = {k: clip[k] for k in ("title", "description", "hashtags", "hook", "start", "end", "score")}
        (d / "clips" / f"{clip['id']}.json").write_text(json.dumps(post, indent=1), encoding="utf-8")
        if clip["version"] == 1:
            clip.setdefault("made_at", now_ts)
            creators.log(job["creator"], f"Clip ready (score {clip['score']}): {clip['title'] or clip['hook']}")
            save(job)
            for fn in on_clip_ready:
                try:
                    fn(job, clip)
                except Exception:  # noqa: BLE001
                    traceback.print_exc()
    except Exception as e:  # noqa: BLE001
        clip.update(status="error", error=safe.friendly(e, 400))
        creators.log(job["creator"], f"Clip failed: {safe.friendly(e, 200)}")
    save(job)


def _caption_words(job, clip, words, off):
    out_lang = brain.clean_output_lang(clip.get("output_lang") or (job.get("options") or {}).get("output_lang") or config.OUTPUT_LANG)
    span = [round(clip["start"], 2), round(clip["end"], 2), f"lang:{out_lang}"]
    if clip.get("fixed_for") == span and clip.get("fixed_words") and not any(
            roman.needs_roman(w.get("w", "")) for w in clip["fixed_words"]):
        raw = clip["fixed_words"]          # already tidied for this exact span and language: no second AI call
    else:
        raw = [w for w in words if clip["start"] - 1 <= w["s"] <= clip["end"] + 1]
        try:
            usage.set_context(job.get("creator"), job.get("title") or "")
            brain._caption_ctx.degraded = False
            raw = brain.english_captions(raw, job.get("lang") or "", job.get("title") or "",
                                         youtube_english=lambda: _youtube_part(job, clip, out_lang),
                                         output_lang=out_lang)
            if not getattr(brain._caption_ctx, "degraded", False):
                clip.update(fixed_words=raw, fixed_for=span)
        except Exception:  # noqa: BLE001 - captions still render with the words as heard
            pass
    # the last word on it: whatever was said, nothing is drawn in Hindi or Urdu script
    return [dict(w, s=w["s"] - off, e=w["e"] - off) for w in roman.words(raw)]


CLIP_MINUTES = float(config.env("PEAKCUT_CLIP_MINUTES", "15"))     # the longest one clip may take to make


def _wait_for_clips(job, futures, began, workers):
    """Wait for a video's clips, but not for ever. A clip that hangs (a download that stalls, an encoder that
    freezes) used to keep the whole video "in progress", and no other clip could be re-made. Now it is marked as
    failed after CLIP_MINUTES and the video finishes with the clips that did work."""
    left, given_up = list(futures), 0
    late = "This clip took too long and was stopped. Open it and press Save and re-make to try again."
    while left:
        still = []
        for f, clip in left:
            if f.done():
                f.result()
                continue
            t = began.get(clip["id"])
            if (t and time.time() - t > CLIP_MINUTES * 60) or (not t and given_up >= workers):
                given_up += 1
                clip.update(status="error", error=late)
                creators.log(job["creator"], f"Clip stopped after {CLIP_MINUTES:g} minutes: {clip.get('title') or clip.get('hook') or clip['id']}")
                save(job)
                continue
            still.append((f, clip))
        left = still
        if left:
            time.sleep(1)


def _youtube_part(job, clip, target_lang: str = "en"):
    """YouTube's captions in `target_lang` ('pt' or 'en') for just this clip, fetched once per video."""
    if (job.get("source") or {}).get("type") != "url" or target_lang not in ("pt", "en"):
        return None
    f = Path(job["dir"]) / f"words_{target_lang}.json"
    if f.is_file():
        tr = json.loads(f.read_text(encoding="utf-8"))
    else:
        try:
            tr = media.youtube_target_words(job["source"]["url"], target_lang) or []
        except Exception:  # noqa: BLE001 - YouTube often refuses its translations: the AI translates instead
            tr = []
        f.write_text(json.dumps(tr), encoding="utf-8")       # asked once per video, refused or not
    return [w for w in tr if clip["start"] - 0.5 <= w["s"] <= clip["end"] + 0.5]


def _english_part(job, clip):
    return _youtube_part(job, clip, "en")


LONG_VIDEO = 40 * 60   # above this, fetch only the pieces that become clips
PAD = 8                # seconds kept either side of a fetched piece, so clips can be re-trimmed


def _piece_again(job, clip):
    """Remaking a clip whose own piece of the video was never fetched (its first try failed) and whose whole video
    is not here: fetch the piece now, cut exactly at the asked time, so no sound matching is needed."""
    src = job.get("source") or {}
    if src.get("type") != "url" or not src.get("url"):
        raise RuntimeError("The original video is no longer here, so this clip can't be made again. Add the video again.")
    d = Path(job["dir"])
    (d / "clips").mkdir(parents=True, exist_ok=True)
    want = max(0.0, clip["start"] - PAD)
    until = clip["end"] + PAD if not job.get("duration") else min(job["duration"], clip["end"] + PAD)
    try:
        piece = media.download_section(src["url"], d / "clips", f"{clip['id']}.src", want, until,
                                       int(job["options"]["quality"]), exact=True)
        meta = media.probe(piece)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Could not fetch this part of the video: {safe.friendly(e, 250)}") from e
    clip.update(src=str(piece), src_offset=round(want, 2), src_end=round(want + meta["duration"], 2))
    clip["end"] = min(clip["end"], clip["src_end"])
    job.setdefault("width", meta["width"])
    job.setdefault("height", meta["height"])


def _fetch_piece(job, clip, full_audio):
    """Long videos: download just this clip's stretch and pin down exactly where it starts."""
    d = Path(job["dir"])
    o = job["options"]
    try:
        want = max(0.0, clip["start"] - PAD)
        piece = media.download_section(job["source"]["url"], d / "clips", f"{clip['id']}.src", want,
                                       min(job["duration"], clip["end"] + PAD), int(o["quality"]))
        wav = d / "clips" / f"{clip['id']}.wav"
        media.extract_audio(piece, wav)
        part = media.load_audio(wav)
        wav.unlink(missing_ok=True)
        meta = media.probe(piece)
        off = media.find_offset(full_audio, part, want)
        clip.update(src=str(piece), src_offset=round(off, 2), src_end=round(off + meta["duration"], 2))
        clip["start"], clip["end"] = max(clip["start"], clip["src_offset"]), min(clip["end"], clip["src_end"])
        job.setdefault("width", meta["width"])
        job.setdefault("height", meta["height"])
    except Exception as e:  # noqa: BLE001
        clip.update(status="error", error=f"Could not fetch this part of the video: {safe.friendly(e, 250)}")


BLOCK = 10 * 60         # long videos are transcribed and clipped one block at a time
PROGRESSIVE_FROM = 20 * 60


def _covers(words: list, start: float, end: float) -> bool:
    """Is a borrowed transcript good enough to find clips with? It has to reach the end of the part being clipped
    and have a normal amount of talking in it; otherwise our own model writes the transcript."""
    inside = [w for w in words if start <= w["s"] <= end]
    minutes = max(0.5, (end - start) / 60)
    return len(inside) >= 25 * minutes and inside[-1]["s"] >= start + 0.85 * (end - start)


def _facecam(job) -> list | None:
    """Where the webcam is in the source: once for the whole video, for every clip of it.

    Only a sure answer is used. A box that is a scoreboard drawn over the game would
    put a picture of a scoreboard in the corner of the clip. One frame is also shown to
    the AI, which tells a person from a game character better than any measurement can;
    what it says is measured again on the picture, so it cannot do any harm."""
    path = job.get("source_path")
    if not path or not Path(path).is_file():
        return None
    try:
        box = facecam.detect(path, ask=brain.look_for_facecam)
    except Exception:  # noqa: BLE001 - a missing facecam is not a failure
        return None
    if not box or box.get("conf") != "high":
        return None
    job["facecam"] = [box["x"], box["y"], box["w"], box["h"]]
    return job["facecam"]


def _save_loudness(d: Path, lo_s: float, db):
    try:
        vals = [round(float(x), 2) for x in db]
        (d / "loudness.json").write_text(json.dumps({"offset": round(float(lo_s), 2), "db": vals}), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def _load_loudness(d: Path, lo_s: float, hi_s: float):
    span_len = max(1, int(math.ceil(max(0.0, hi_s - lo_s))) + 1)
    lf = d / "loudness.json"
    if lf.is_file():
        try:
            raw = json.loads(lf.read_text(encoding="utf-8"))
            off = float(raw.get("offset", 0.0))
            arr = raw.get("db") or []
            if lo_s >= off and arr:
                s0 = int(lo_s - off)
                sliced = arr[s0: s0 + span_len]
                if len(sliced) >= min(span_len, 10):
                    out = np.zeros(span_len, dtype=np.float32)
                    n = min(len(sliced), span_len)
                    out[:n] = np.asarray(sliced[:n], dtype=np.float32)
                    return out
        except Exception:  # noqa: BLE001
            pass
    return np.zeros(span_len, dtype=np.float32)


def _queue_found_clips(job, found, shift: float, o: dict, statics: list, notes: list, face,
                       renders, futures, make_fn, words: list) -> tuple[list, object]:
    if found and statics:
        keep = []
        for c in found:
            a_s, b_s = c["start"] + shift, c["end"] + shift
            inside = quality.static_overlap(a_s, b_s, statics)
            if inside > 0.5 * max(1.0, b_s - a_s):
                notes.append(f"A clip from {round(a_s)}s was left out: it sat in an ad break or a static screen.")
            else:
                keep.append(c)
        found = keep
    created = []
    is_16_9 = o.get("layout") == "horizontal"
    for c in found:
        cs, ce = round(c["start"] + shift, 2), round(c["end"] + shift, 2)
        c.update(
            id=_next_id(job, "clip"),
            start=cs,
            end=ce,
            style=o["style"],
            layout=o["layout"],
            captions=(False if is_16_9 else o.get("captions", True)),
            output_lang=o.get("output_lang", config.OUTPUT_LANG),
            thumb_mode=o.get("thumb_mode", "local"),
            status="queued",
            version=0,
        )
        if is_16_9:
            c["format"] = "16:9"
            c["peak_t"] = round((cs + ce) / 2, 2)
        if o["layout"] == "gameplay" and face is False:
            face = _facecam(job)
        if face:
            c["facecam"] = face
        job["clips"].append(c)
        created.append(c)
        futures.append((renders.submit(ctx.bind(make_fn), c, list(words)), c))
    return created, face


def _run_job(job):
    from concurrent.futures import ThreadPoolExecutor
    d = Path(job["dir"])
    o = job["options"]
    src = job["source"]
    # Which game this is: what the creator set, or what the title says. It decides the extra rules the AI
    # reads and the tags every post carries.
    game = job.get("game") or ""
    if not game and job.get("creator"):
        try:
            game = (creators.get(job["creator"]) or {}).get("game") or ""
        except Exception:  # noqa: BLE001
            game = ""
    if not game:
        game = games.detect(job["title"], src.get("name"))[0] or ""
    if game and job.get("game") != game:
        job["game"] = game
    t0 = time.time()
    notes = []
    face = False       # where the webcam is in the source, found once; False means not looked for yet
    usage.set_context(job["creator"], job["title"] or src.get("name"))
    resuming = bool(job.pop("resume", False)) and (d / "words.json").is_file()
    if resuming:   # after a restart: keep the transcript parts, clips and AI answers already made, and carry on
        _set(job, status="running", stage="Picking up where it stopped", error=None)
    else:
        _set(job, status="running", stage="Getting the audio", progress=0.03, error=None, clips=[], ai={}, done_blocks=0)
    if not transcribe.model_ready() and not transcribe.REMOTE:
        from . import fetch_model
        _set(job, stage="Getting the speech model (one time, about 480 MB)")
        fetch_model.ensure()
        _set(job, stage="Getting the audio")
    fetch = ThreadPoolExecutor(1)
    video = None
    audio = None
    cached_job = None
    pieces = False
    if src["type"] == "url":
        meta = media.info(src["url"])
        if meta["kind"] != "video":
            raise RuntimeError("That link is a channel or playlist, not a single video. Paste the link of one video.")
        if ((o["range_end"] or meta["duration"]) - (o["range_start"] or 0)) > config.MAX_VIDEO_HOURS * 3600:
            raise RuntimeError(f"That is longer than {config.MAX_VIDEO_HOURS:g} hours. Choose a shorter part of it under Options.")
        if meta["is_live"]:
            raise RuntimeError("This link is a live stream, so it would never finish downloading. Open the creator, "
                               "put this link under Setup, and press Start recording instead.")
        _set(job, title=meta["title"], uploader=meta["uploader"], duration=meta["duration"])
        if not job.get("thumb") and media.save_thumb(d, url=src["url"]):
            _set(job, thumb=True)
        usage.set_context(job["creator"], meta["title"])
        lo_s = float(o["range_start"] or 0.0)
        hi_s = float(o["range_end"] or meta["duration"] or 0.0)
        if not resuming and hi_s > lo_s:
            cached_job = _find_cached_job(job, lo_s, hi_s)
        span = hi_s - lo_s
        pieces = bool(cached_job) or span > LONG_VIDEO or meta["duration"] <= 0
        if not cached_job:
            if not pieces:  # shorter video: fetch all of it in the background while the audio is worked on
                video = fetch.submit(media.download, src["url"], d, int(o["quality"]))
            audio_file = media.download_audio(src["url"], d, lambda p: _set(job, progress=0.03 + 0.1 * p))
    else:
        pieces, audio_file = False, Path(src["path"])
        info = media.probe(audio_file)
        if info["duration"] > config.MAX_VIDEO_HOURS * 3600:
            raise RuntimeError(f"That video is longer than {config.MAX_VIDEO_HOURS:g} hours, which is the limit.")
        _set(job, source_path=str(audio_file), width=info["width"], height=info["height"], duration=info["duration"])
        if not job.get("thumb") and media.save_thumb(d, video=audio_file):
            _set(job, thumb=True)
    if cached_job:
        cdir = Path(cached_job["dir"])
        words = json.loads((cdir / "words.json").read_text(encoding="utf-8"))
        (d / "words.json").write_text(json.dumps(words), encoding="utf-8")
        if cached_job.get("lang"):
            job["lang"] = cached_job["lang"]
        job["transcript_from"] = "cache"
        lo_s = float(o["range_start"] or 0.0)
        hi_s = min(float(job.get("duration") or words[-1]["e"]), float(o["range_end"] or job.get("duration") or words[-1]["e"]))
        lo, hi = int(lo_s * 16000), int(hi_s * 16000)
        span = max(0.0, hi_s - lo_s)
        db = _load_loudness(cdir, lo_s, hi_s)
        _save_loudness(d, lo_s, db)
    else:
        wav = d / "audio.wav"
        media.extract_audio(audio_file, wav)
        audio = media.load_audio(wav)
        wav.unlink(missing_ok=True)
        if src["type"] == "url":
            audio_file.unlink(missing_ok=True)
        if not job["duration"]:
            job["duration"] = len(audio) / 16000
        lo = int((o["range_start"] or 0) * 16000)
        hi = min(len(audio), int(o["range_end"] * 16000)) if o["range_end"] else len(audio)
        span = (hi - lo) / 16000
        db = media.loudness_per_second(audio[lo:hi])
        _save_loudness(d, lo / 16000, db)

    vlock, vstate = threading.Lock(), {}

    def whole_video():
        """Wait for the background download once; retry it, or give up and use pieces."""
        with vlock:
            if "path" not in vstate:
                path = None
                try:
                    path = video.result()
                except Exception as e:  # noqa: BLE001 - sites sometimes refuse a download
                    notes.append(f"Video download refused once ({str(e)[-80:]}), retrying")
                    try:
                        path = media.download(src["url"], d, int(o["quality"]))
                    except Exception as e2:  # noqa: BLE001
                        notes.append(f"Whole-video download failed ({str(e2)[-80:]}), fetching clip parts instead")
                if path:
                    info = media.probe(path)
                    job.update(source_path=str(path), width=info["width"], height=info["height"])
                vstate["path"] = path
            return vstate["path"]

    exact = {}      # clip id -> (from, to, words): our own model's words for just that clip

    began = {}      # clip id -> when work on it started
    yt = None

    def make(clip, words_now):
        began[clip["id"]] = time.time()
        if src["type"] == "url" and not _reuse_cached_src(job, clip) and (pieces or audio is None or not whole_video()):
            if audio is not None and len(audio):
                _fetch_piece(job, clip, audio)
            else:
                try:
                    _piece_again(job, clip)
                except Exception as e:  # noqa: BLE001
                    clip.update(status="error", error=safe.friendly(e, 250))
        if audio is not None and clip["status"] != "error" and clip.get("captions", True) and yt and yt[1].lower().startswith(("en", "pt")):
            # The clip was found with YouTube's transcript. What is shown on screen comes from our own model,
            # run on just this clip: full punctuation, nothing blanked out, timings to the word.
            a0, b0 = max(lo, int((clip["start"] - 1.5) * 16000)), min(hi, int((clip["end"] + 1.5) * 16000))
            try:
                tw = [dict(w, s=round(w["s"] + a0 / 16000, 2), e=round(w["e"] + a0 / 16000, 2)) for w in transcribe.transcribe(audio[a0:b0])]
            except Exception:  # noqa: BLE001 - the clip is still made, with YouTube's words
                tw = []
            if len(tw) >= 3:
                exact[clip["id"]] = (a0 / 16000, b0 / 16000, tw)
                words_now = tw
        if clip["status"] != "error":
            render_one(job, clip, words_now)

    cpus = os.cpu_count() or 4
    renders = ThreadPoolExecutor(2 if cpus < 8 else min(6, cpus // 6))  # more clips made at once on a bigger machine
    futures = []
    ahead = ThreadPoolExecutor(2, thread_name_prefix="ahead")
    tokens, model = 0, None
    statics = quality.static_ranges(job.get("source_path") or "") if config.SKIP_STATIC else []

    if cached_job:
        _set(job, stage="Using cached transcript to find clips", progress=0.55, word_count=len(words))
        lo_s = lo / 16000
        hi_s = hi / 16000
        bw = [dict(w, s=round(w["s"] - lo_s, 2), e=round(w["e"] - lo_s, 2)) for w in words if lo_s <= w["s"] <= hi_s]
        if len(bw) >= 8:
            found, ai = brain.find_clips(bw, db, span, job["title"], job["uploader"], int(o["max_clips"]),
                                         int(o["min_len"]), int(o["max_len"]), notes.append, game=game,
                                         output_lang=o.get("output_lang", config.OUTPUT_LANG))
            tokens += ai.get("tokens", 0)
            model = ai.get("model") or model
            if ai.get("note"):
                notes.append(ai["note"])
            _, face = _queue_found_clips(job, found, lo_s, o, statics, notes, face, renders, futures, make, words)
        job["ai"] = {"model": model, "tokens": tokens, "note": "", "log": notes[-5:]}
        save(job)
    else:
        step = BLOCK * 16000 if span > PROGRESSIVE_FROM else hi - lo
        blocks = [(x, min(hi, x + step)) for x in range(lo, hi, step)] or [(lo, hi)]
        words = []
        done_blocks = 0
        if src["type"] == "url" and config.YT_TRANSCRIPT and media.is_youtube(src["url"]):
            try:
                yt = media.youtube_words(src["url"])
            except Exception:  # noqa: BLE001 - no transcript to borrow: ours is written as usual
                yt = None
            if yt and not _covers(yt[0], lo / 16000, hi / 16000):
                yt = None
            job["transcript_from"] = "youtube" if yt else "own"
            job["lang"] = yt[1] if yt else ""          # the language spoken, when YouTube says (decides the captions)
        if resuming:
            try:
                words = json.loads((d / "words.json").read_text(encoding="utf-8"))
            except ValueError:
                words = []
            done_blocks = min(int(job.get("done_blocks", 0)), len(blocks)) if words else 0
            tokens, model = int((job.get("ai") or {}).get("tokens", 0)), (job.get("ai") or {}).get("model")
            if done_blocks < len(blocks):     # words past the last finished part are transcribed again
                cut = blocks[done_blocks][0] / 16000
                words = [w for w in words if w["s"] < cut]
            for c in job["clips"]:            # clips found before the restart that were not finished yet
                if c.get("status") not in ("done", "error"):
                    c["status"] = "queued"
                    futures.append((renders.submit(ctx.bind(make), c, list(words)), c))
            save(job)
        heard, now_on = {}, [done_blocks]

        def hear(i):
            if i < len(blocks) and i >= done_blocks and i not in heard:
                a_, b_ = blocks[i]
                last = [0.0]

                def tp(p, i=i, a_=a_, b_=b_):
                    if now_on[0] == i and p - last[0] > 0.05:      # only the part being shown moves the bar
                        last[0] = p
                        _set(job, progress=max(job["progress"], 0.14 + 0.78 * ((a_ - lo) + p * (b_ - a_)) / max(1, hi - lo)))
                heard[i] = ahead.submit(ctx.bind(transcribe.transcribe), audio[a_:b_], tp)

        for bi, (a, b) in enumerate(blocks):
            if bi < done_blocks:
                continue                      # transcribed and checked by the AI before the restart
            shift = a / 16000
            where = f" (minute {round((a - lo) / 16000 / 60)} of {round(span / 60)})" if len(blocks) > 1 else ""
            if yt:
                _set(job, stage="Reading the transcript" + where)
                bw = [dict(w, s=round(w["s"] - shift, 2), e=round(w["e"] - shift, 2)) for w in yt[0] if a / 16000 <= w["s"] < b / 16000]
            else:
                _set(job, stage="Writing the transcript" + where)
                now_on[0] = bi
                hear(bi)
                hear(bi + 1)
                bw = heard.pop(bi).result()
            _set(job, progress=max(job["progress"], 0.14 + 0.78 * (b - lo) / max(1, hi - lo)))
            words += [dict(w, s=round(w["s"] + shift, 2), e=round(w["e"] + shift, 2)) for w in bw]
            (d / "words.json").write_text(json.dumps(words), encoding="utf-8")
            _set(job, stage="Finding the best moments" + where, word_count=len(words))
            left = int(o["max_clips"]) - len(job["clips"])
            if left <= 0 or len(bw) < 8:
                job["done_blocks"] = bi + 1
                save(job)
                continue
            quota = max(1, math.ceil(left / (len(blocks) - bi)))
            s0 = int((a - lo) / 16000)
            found, ai = brain.find_clips(bw, db[s0: s0 + int((b - a) / 16000) + 1], (b - a) / 16000, job["title"],
                                         job["uploader"], quota, int(o["min_len"]), int(o["max_len"]), notes.append, game=game,
                                         output_lang=o.get("output_lang", config.OUTPUT_LANG))
            tokens += ai.get("tokens", 0)
            model = ai.get("model") or model
            if ai.get("note"):
                notes.append(ai["note"])
            _, face = _queue_found_clips(job, found, shift, o, statics, notes, face, renders, futures, make, words)
            job["ai"] = {"model": model, "tokens": tokens, "note": "", "log": notes[-5:]}
            job["done_blocks"] = bi + 1       # a restart from here on carries on with the next part
            save(job)
    long_enq, long_tok, long_model = _apply_long_and_derived(job, words, db, span, lo, game, notes, face=face, audio=audio)
    tokens += long_tok
    model = long_model or model
    for c in long_enq:
        futures.append((renders.submit(ctx.bind(make), c, list(words)), c))
    if long_enq:
        save(job)
    if futures:
        _set(job, stage="Finishing the clips", progress=max(job["progress"], 0.92))
    _wait_for_clips(job, futures, began, renders._max_workers)
    if exact:           # keep our own words for the clips in the saved transcript, so editing a clip later uses them
        for a_s, b_s, tw in exact.values():
            words = [w for w in words if not a_s <= w["s"] < b_s] + tw
        words.sort(key=lambda w: w["s"])
        (d / "words.json").write_text(json.dumps(words), encoding="utf-8")
    renders.shutdown(wait=False)
    ahead.shutdown(wait=False)
    fetch.shutdown(wait=False)
    del audio
    clips = job["clips"]
    job["ai"] = {"model": model, "tokens": tokens, "log": notes[-5:],
                 "note": "The AI was unavailable, so clips were picked by loudness only." if clips and not model else ""}
    stage = "Done" if clips else "Done, but nothing clip-worthy was found"
    _set(job, status="done", stage=stage, progress=1.0, took=round(time.time() - t0))
    try:
        from . import notices
        name = job.get("title") or job["source"].get("name") or "your video"
        notices.add(job.get("user"), "clips", f"{len(clips)} clip{'s' if len(clips) != 1 else ''} ready" if clips else "Nothing clip-worthy found",
                    name, f"/app/creators/{job['creator']}" if job.get("creator") else "/app/videos")
    except Exception:  # noqa: BLE001
        pass
    creators.log(job["creator"], f"Video finished with {len(clips)} clips in {round(time.time() - t0)}s: {job['title']}")


def _span_has_music(words, audio, start: float, end: float, candidate: dict | None = None) -> tuple[bool, str]:
    if candidate and candidate.get("has_music"):
        return True, str(candidate.get("music_reason") or "Música (>5s) detectada pela IA")
    res = quality.detect_music(words=words, audio=audio, start=start, end=end, min_seconds=5.0)
    return bool(res.get("has_music")), str(res.get("reason") or "")


def _spawn_shorts_for_span(job, words, db, lo, span_start: float, span_end: float, base_title: str,
                           base_hashtags: list, game, notes: list, face=None, reason: str = "", opts: dict | None = None) -> tuple[list, int, str | None]:
    """Create 9:16 Shorts/Reels from a segment blocked from long upload due to >5s music."""
    o = opts or job["options"]
    out_lang = o.get("output_lang", config.OUTPUT_LANG)
    shift_lo = lo / 16000
    max_short_len = min(60, max(15, int(o.get("max_len", 60))))
    min_short_len = min(max_short_len - 5, max(10, int(o.get("min_len", 15))))
    short_cap = min(3, max(1, int(o.get("max_clips", 6))))
    existing = [
        c for c in job["clips"]
        if c.get("format", "9:16") != "16:9" and c.get("clip_kind") != "long"
        and c["start"] >= span_start - 2 and c["end"] <= span_end + 2
    ]
    if existing:
        for c in existing:
            c["has_music"] = True
        return [], 0, None
    bw = [
        dict(w, s=round(w["s"] - span_start, 2), e=round(w["e"] - span_start, 2))
        for w in words if span_start <= w["s"] <= span_end
    ]
    created_raw, tok, model = [], 0, None
    if len(bw) >= 8:
        s0 = max(0, int(span_start - shift_lo))
        s1 = min(len(db), int(span_end - shift_lo) + 1)
        found, ai_sub = brain.find_clips(
            bw, db[s0:s1], span_end - span_start, job.get("title", ""), job.get("uploader", ""),
            short_cap, min_short_len, max_short_len, notes.append, game=game, output_lang=out_lang,
        )
        tok += ai_sub.get("tokens", 0)
        model = ai_sub.get("model") or model
        for sc in found:
            sc["start"] = round(sc["start"] + span_start, 2)
            sc["end"] = round(sc["end"] + span_start, 2)
            created_raw.append(sc)
    if not created_raw and bw:
        parts = brain.split_series_parts(
            words, span_start, span_end, base_title=base_title, base_hashtags=base_hashtags,
            part_min=min(40, max_short_len - 10), part_max=max_short_len,
            score=80, game=game, output_lang=out_lang,
        )[:short_cap]
        created_raw.extend(parts)
    enqueued = []
    for sc in created_raw[:short_cap]:
        sc.update(
            id=_next_id(job, "clip"),
            style=o["style"],
            layout=o["layout"],
            format="9:16",
            has_music=True,
            music_reason=reason,
            captions=o.get("captions", True),
            output_lang=out_lang,
            thumb_mode=o.get("thumb_mode", "local"),
            status="queued",
            version=0,
        )
        if face:
            sc["facecam"] = face
        job["clips"].append(sc)
        enqueued.append(sc)
    return enqueued, tok, model


def _apply_long_and_derived(job, words, db, span, lo, game, notes, face=None, audio=None, opts: dict | None = None, exclude_long=()):
    """Generate 16:9 long blocks, derived 9:16 Shorts (parent_clip), and 16:9 compilations after blocks finish."""
    o = opts or job["options"]
    long_mode = o.get("long_mode", "off")
    if long_mode == "off":
        return [], 0, None
    long_min_len = int(o.get("long_min_len", 600))
    long_max_len = int(o.get("long_max_len", 1200))
    out_lang = o.get("output_lang", config.OUTPUT_LANG)
    shift_lo = lo / 16000
    enqueued, tokens, model = [], 0, None
    rel_excl = [
        {"start": max(0.0, float(x["start"]) - shift_lo), "end": max(0.0, float(x["end"]) - shift_lo)}
        for x in (exclude_long or ())
        if float(x.get("end", 0)) > shift_lo
    ]
    if long_mode in ("block", "both") and span >= long_min_len and len(words) >= 20:
        shifted_words = (
            [dict(w, s=round(w["s"] - shift_lo, 2), e=round(w["e"] - shift_lo, 2)) for w in words]
            if shift_lo else words
        )
        blocks, ai_long = brain.find_long_blocks(
            shifted_words, db, span, job["title"], job["uploader"],
            max_blocks=2, min_len=long_min_len, max_len=long_max_len,
            log=notes.append, game=game, output_lang=out_lang, exclude=rel_excl,
        )
        tokens += ai_long.get("tokens", 0)
        model = ai_long.get("model") or model
        if ai_long.get("note"):
            notes.append(ai_long["note"])
        for b in blocks:
            abs_s = round(b["start"] + shift_lo, 2)
            abs_e = round(b["end"] + shift_lo, 2)
            has_music, music_why = _span_has_music(words, audio, abs_s, abs_e, b)
            if has_music:
                msg = f"Bloco longo ({round(abs_s)}-{round(abs_e)}s) contém música >5s ({music_why or 'áudio musical'}): convertido apenas em Shorts/Reels."
                notes.append(msg)
                creators.log(job["creator"], msg)
                sp_enq, sp_tok, sp_mod = _spawn_shorts_for_span(
                    job, words, db, lo, abs_s, abs_e,
                    b.get("title") or job.get("title", ""), b.get("hashtags") or [],
                    game, notes, face=face, reason=music_why, opts=o,
                )
                enqueued.extend(sp_enq)
                tokens += sp_tok
                model = sp_mod or model
                continue
            b.update(
                id=_next_id(job, "long"),
                start=abs_s,
                end=abs_e,
                peak_t=round(b.get("peak_t", b["start"]) + shift_lo, 2),
                style="clean",
                layout="horizontal",
                format="16:9",
                clip_kind="long",
                long_type="block",
                has_music=False,
                captions=False,
                output_lang=out_lang,
                thumb_mode=o.get("thumb_mode", "local"),
                status="queued",
                version=0,
            )
            if face:
                b["facecam"] = face
            job["clips"].append(b)
            enqueued.append(b)
            if o.get("derive_shorts", True):
                short_cap = min(3, int(o.get("max_clips", 6)))
                inside = [
                    c for c in job["clips"]
                    if c.get("format", "9:16") != "16:9" and c.get("clip_kind") != "long"
                    and not c.get("parent_clip")
                    and c["start"] >= b["start"] - 2 and c["end"] <= b["end"] + 2
                ]
                for c in inside[:short_cap]:
                    c["parent_clip"] = b["id"]
                if not inside:
                    bw = [
                        dict(w, s=round(w["s"] - b["start"], 2), e=round(w["e"] - b["start"], 2))
                        for w in words if b["start"] <= w["s"] <= b["end"]
                    ]
                    if len(bw) >= 8:
                        s0 = max(0, int(b["start"] - shift_lo))
                        s1 = min(len(db), int(b["end"] - shift_lo) + 1)
                        sub_clips, ai_sub = brain.find_clips(
                            bw, db[s0:s1], b["end"] - b["start"], job["title"], job["uploader"],
                            min(2, short_cap), int(o.get("min_len", 15)), int(o.get("max_len", 60)),
                            notes.append, game=game, output_lang=out_lang,
                        )
                        tokens += ai_sub.get("tokens", 0)
                        model = ai_sub.get("model") or model
                        for sc in sub_clips:
                            sc.update(
                                id=_next_id(job, "clip"),
                                start=round(sc["start"] + b["start"], 2),
                                end=round(sc["end"] + b["start"], 2),
                                style=o["style"],
                                layout=o["layout"],
                                format="9:16",
                                parent_clip=b["id"],
                                captions=o.get("captions", True),
                                output_lang=out_lang,
                                thumb_mode=o.get("thumb_mode", "local"),
                                status="queued",
                                version=0,
                            )
                            if face:
                                sc["facecam"] = face
                            job["clips"].append(sc)
                            enqueued.append(sc)
    if long_mode in ("compile", "both"):
        short_cands = []
        for c in job["clips"]:
            if c.get("format", "9:16") != "16:9" and c.get("clip_kind") != "long" and c.get("status") != "error":
                hm, hw = _span_has_music(words, audio, c["start"], c["end"], c)
                if hm:
                    c["has_music"] = True
                    if hw:
                        c["music_reason"] = hw
                    continue
                short_cands.append(c)
        comp = brain.build_compilation(
            short_cands, title=job["title"], uploader=job["uploader"],
            min_len=long_min_len, max_len=long_max_len, game=game, output_lang=out_lang,
        )
        if comp:
            for seg in comp.get("segments") or []:
                match = next(
                    (
                        c for c in short_cands
                        if c.get("src") and Path(c["src"]).is_file()
                        and float(c.get("src_offset", c["start"])) <= float(seg["start"]) + 0.05
                        and float(c.get("src_end", c["end"])) >= float(seg["end"]) - 0.05
                    ),
                    None,
                )
                if match:
                    seg["src"] = match["src"]
                    seg["src_offset"] = float(match.get("src_offset", 0.0))
            comp.update(
                id=_next_id(job, "comp"),
                style="clean",
                layout="horizontal",
                format="16:9",
                clip_kind="long",
                long_type="compile",
                has_music=False,
                captions=False,
                output_lang=out_lang,
                thumb_mode=o.get("thumb_mode", "local"),
                status="queued",
                version=0,
            )
            if face:
                comp["facecam"] = face
            job["clips"].append(comp)
            enqueued.append(comp)
    return enqueued, tokens, model


def request_recut(jid: str, options: dict | None = None) -> dict:
    """Queue a new round of clips on an already transcribed job using its cached words.json."""
    with _lock:
        job = get(jid)
        if not job:
            raise KeyError(jid)
        if job.get("status") in ("queued", "running", "live"):
            raise RuntimeError("This video is still being worked on.")
        d = Path(job["dir"])
        if not (d / "words.json").is_file():
            raise ValueError("This video has no saved transcript cache yet.")
        has_job_src = bool(job.get("source_path") and Path(job["source_path"]).is_file())
        has_url_src = bool((job.get("source") or {}).get("type") == "url" and (job.get("source") or {}).get("url"))
        if not (has_job_src or has_url_src):
            raise ValueError("The original video source is not available to render new clips.")
        raw = dict(options or {})
        cleaned = clean_options(raw)
        opts = {**DEFAULTS, **(job.get("options") or {}), **cleaned}
        for rk in ("range_start", "range_end"):
            if rk in raw and raw[rk] in (None, ""):
                opts[rk] = None
        if opts.get("range_end") is not None and opts["range_end"] <= (opts.get("range_start") or 0):
            raise ValueError("The end of the chosen part must be after its start.")
        if opts["min_len"] >= opts["max_len"]:
            opts["min_len"] = max(5, opts["max_len"] - 5)
        if opts["long_min_len"] >= opts["long_max_len"]:
            opts["long_min_len"] = max(300, opts["long_max_len"] - 60)
        opts["avoid_overlap"] = bool(raw.get("avoid_overlap", True))
        job["pending_recut"] = opts
        job.update(status="queued", stage="Waiting to make more clips (using cache)", error=None, progress=0.05)
        save(job)
    _q.put(("recut", jid, None), user=job["user"])
    return job


def _run_recut(job):
    from concurrent.futures import ThreadPoolExecutor
    d = Path(job["dir"])
    pending = dict(job.pop("pending_recut", None) or job.get("options") or DEFAULTS)
    avoid_overlap = bool(pending.pop("avoid_overlap", True))
    o = {**DEFAULTS, **(job.get("options") or {}), **pending}
    job["options"] = o
    src = job["source"]
    game = job.get("game") or games.detect(job.get("title", ""), src.get("name"))[0] or ""
    t0 = time.time()
    notes = []
    face = job.get("facecam") if job.get("facecam") else False
    usage.set_context(job.get("creator"), job.get("title") or src.get("name"))
    _set(job, status="running", stage="Loading cached transcript", progress=0.1, error=None)

    words = json.loads((d / "words.json").read_text(encoding="utf-8"))
    total_dur = float(job.get("duration") or (words[-1]["e"] if words else 0.0) or 0.0)
    if total_dur <= 0 and words:
        total_dur = float(words[-1]["e"])
        job["duration"] = total_dur
    lo_s = float(o.get("range_start") or 0.0)
    hi_s = float(o.get("range_end") or total_dur)
    hi_s = min(total_dur, hi_s) if total_dur > 0 else hi_s
    span = max(0.0, hi_s - lo_s)
    db = _load_loudness(d, lo_s, hi_s)

    want_16_9 = o.get("layout") == "horizontal"
    exclude = []
    if avoid_overlap:
        for c in job.get("clips", []):
            if c.get("status") == "error":
                continue
            c_16_9 = c.get("format") == "16:9" or c.get("layout") == "horizontal" or c.get("clip_kind") == "long"
            if c_16_9 == want_16_9:
                exclude.append({"start": float(c["start"]), "end": float(c["end"])})

    cpus = os.cpu_count() or 4
    renders = ThreadPoolExecutor(2 if cpus < 8 else min(6, cpus // 6))
    futures = []
    began = {}
    tokens, model = 0, None
    statics = quality.static_ranges(job.get("source_path") or "") if config.SKIP_STATIC else []

    def make(clip, words_now):
        began[clip["id"]] = time.time()
        render_one(job, clip, words_now)

    max_len_i = int(o.get("max_len", 60))
    step_sec = span if (span <= PROGRESSIVE_FROM or max_len_i > 180) else max(float(BLOCK), float(max_len_i) * 3.0)
    step_sec = max(1.0, step_sec)
    blocks = []
    cur = lo_s
    while cur < hi_s - 0.5:
        nxt = min(hi_s, cur + step_sec)
        blocks.append((cur, nxt))
        if nxt >= hi_s:
            break
        cur = nxt
    if not blocks:
        blocks = [(lo_s, hi_s)]

    new_clips = []
    target_new = int(o.get("max_clips", 6))
    for bi, (a_s, b_s) in enumerate(blocks):
        left = target_new - len(new_clips)
        if left <= 0:
            break
        where = f" (part {bi + 1} of {len(blocks)})" if len(blocks) > 1 else ""
        _set(job, stage="Finding new moments from cache" + where,
             progress=max(job["progress"], 0.15 + 0.55 * (bi / max(1, len(blocks)))))
        bw = [dict(w, s=round(w["s"] - a_s, 2), e=round(w["e"] - a_s, 2)) for w in words if a_s <= w["s"] <= b_s]
        if len(bw) < 8:
            continue
        quota = max(1, math.ceil(left / (len(blocks) - bi)))
        s0 = max(0, int(a_s - lo_s))
        s1 = min(len(db), s0 + int(b_s - a_s) + 1)
        rel_excl = [
            {"start": x["start"] - a_s, "end": x["end"] - a_s}
            for x in (exclude + new_clips)
            if x["end"] > a_s and x["start"] < b_s
        ]
        found, ai_res = brain.find_clips(
            bw, db[s0:s1], b_s - a_s, job.get("title", ""), job.get("uploader", ""),
            quota, int(o["min_len"]), int(o["max_len"]), notes.append, game=game,
            output_lang=o.get("output_lang", config.OUTPUT_LANG), exclude=rel_excl,
        )
        tokens += ai_res.get("tokens", 0)
        model = ai_res.get("model") or model
        if ai_res.get("note"):
            notes.append(ai_res["note"])
        added, face = _queue_found_clips(job, found[:left], a_s, o, statics, notes, face, renders, futures, make, words)
        new_clips.extend(added)
        save(job)

    long_excl = [
        {"start": float(c["start"]), "end": float(c["end"])}
        for c in job.get("clips", [])
        if c.get("status") != "error" and c.get("clip_kind") == "long"
    ] if avoid_overlap else ()
    long_enq, long_tok, long_model = _apply_long_and_derived(
        job, words, db, span, int(lo_s * 16000), game, notes, face=face, audio=None, opts=o, exclude_long=long_excl,
    )
    tokens += long_tok
    model = long_model or model
    new_clips.extend(long_enq)
    for c in long_enq:
        futures.append((renders.submit(ctx.bind(make), c, list(words)), c))
    if long_enq:
        save(job)
    if futures:
        _set(job, stage="Rendering new clips from cache", progress=0.85)
    _wait_for_clips(job, futures, began, renders._max_workers)
    renders.shutdown(wait=False)
    job["ai"] = {
        "model": model,
        "tokens": int((job.get("ai") or {}).get("tokens", 0)) + tokens,
        "log": notes[-5:],
        "note": "The AI was unavailable, so new clips were picked by loudness only." if new_clips and not model else "",
    }
    n_new = len(new_clips)
    stage = (f"Pronto (+{n_new} {'novo corte' if n_new == 1 else 'novos cortes'} do cache)" if n_new
             else "Nenhum corte novo encontrado nesse trecho (tente outro intervalo ou desmarque evitar repetição)")
    _set(job, status="done", stage=stage, progress=1.0, took=round(time.time() - t0))
    return job


_THUMB_FRACS = (0.50, 0.22, 0.78, 0.35, 0.65, 0.14, 0.86, 0.43, 0.58, 0.28, 0.72)


def regenerate_thumb(jid: str, cid: str, text: str | None = None, mode: str | None = None,
                     peak_t: float | None = None, collage: bool = True, new_variant: bool = True) -> dict:
    """Quickly regenerate a clip's 1280x720 JPEG cover without re-encoding the clip video."""
    job = get(jid)
    if not job:
        raise KeyError(jid)
    clip = next((c for c in job["clips"] if c["id"] == cid), None)
    if not clip:
        raise KeyError(cid)
    if clip.get("status") in ("queued", "rendering"):
        raise RuntimeError("This clip is still being made. Try again when it finishes.")
    d = Path(job["dir"])
    clips_dir = d / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    using = "file"
    src_file = None
    temp_src_for_drive = False
    if clip.get("src") and Path(clip["src"]).is_file():
        src_file = Path(clip["src"])
        using = "src"
    elif job.get("source_path") and Path(job["source_path"]).is_file():
        src_file = Path(job["source_path"])
        using = "job"
    elif clip.get("file") and (clips_dir / Path(str(clip["file"])).name).is_file():
        src_file = clips_dir / Path(str(clip["file"])).name
        using = "file"
    elif (job.get("source") or {}).get("type") == "url" and (job.get("source") or {}).get("url"):
        _piece_again(job, clip)
        if clip.get("src") and Path(clip["src"]).is_file():
            src_file = Path(clip["src"])
            using = "src"
            temp_src_for_drive = bool(clip.get("drive"))
    if not src_file or not src_file.is_file():
        raise ValueError("No video frame source is available on disk to generate this thumbnail.")

    variant = int(clip.get("thumb_variant", 0)) + 1 if new_variant else int(clip.get("thumb_variant", 0))
    cs, ce = float(clip["start"]), float(clip["end"])
    dur = max(0.5, ce - cs)
    alt_abs = None
    if peak_t is not None:
        target_abs = float(peak_t)
    elif new_variant:
        frac = _THUMB_FRACS[(variant - 1) % len(_THUMB_FRACS)]
        cycle = (variant - 1) // len(_THUMB_FRACS)
        frac = min(0.92, max(0.08, frac + (cycle % 4) * 0.03))
        target_abs = cs + dur * frac
        alt_frac = 1.0 - frac
        if abs((alt_frac - frac) * dur) < 1.5 and dur >= 4.0:
            alt_frac = 0.85 if frac < 0.5 else 0.15
        alt_abs = cs + dur * alt_frac
    else:
        target_abs = float(clip.get("peak_t") or ((cs + ce) / 2.0))

    def _to_local(abs_t: float) -> float:
        if using == "src":
            return max(0.0, abs_t - float(clip.get("src_offset", 0.0)))
        if using == "job":
            return max(0.0, abs_t)
        return max(0.0, min(max(0.2, ce - cs) - 0.1, abs_t - cs))

    local_peak_t = _to_local(target_abs)
    local_alt_t = _to_local(alt_abs) if alt_abs is not None else None

    thumb_text = str(text if text is not None else (clip.get("thumb_text") or clip.get("hook") or clip.get("title") or ""))[:40]
    thumb_mode = mode if mode in ("local", "ai") else (clip.get("thumb_mode") or (job.get("options") or {}).get("thumb_mode") or "local")
    face = (clip.get("facecam") or job.get("facecam")) if using in ("src", "job") else None

    render.install_fonts()
    out_jpg = clips_dir / f"{cid}.jpg"
    _, report = thumbs.generate_with_report(
        src_file,
        out_jpg,
        peak_t=local_peak_t,
        text=thumb_text,
        facecam=face,
        mode=thumb_mode,
        collage=bool(collage),
        variant=variant,
        alt_peak_t=local_alt_t,
    )
    clip["thumb_text"] = thumb_text
    clip["thumb_mode"] = thumb_mode
    clip["thumb_variant"] = variant
    clip["thumb_meta"] = report
    clip["cover"] = f"{cid}.jpg"
    clip["peak_t"] = round(target_abs, 2)
    clip["version"] = int(clip.get("version", 0)) + 1
    if temp_src_for_drive and src_file:
        try:
            src_file.unlink(missing_ok=True)
        except OSError:
            pass
        clip.pop("src", None)
        clip.pop("src_offset", None)
        clip.pop("src_end", None)
    save(job)
    return job


def _worker(quick_only=False):
    while True:
        kind, a, cid = _q.get(quick_only)
        if kind == "fn":
            try:
                a()
            except Exception:  # noqa: BLE001
                traceback.print_exc()
            continue
        job = _jobs.get(a)
        if not job:
            if kind in ("job", "recut"):
                _q.done()
            continue
        token = ctx._uid.set(job["user"])  # everything this job does runs as its owner
        try:
            if kind == "job":
                _run_job(job)
            elif kind == "recut":
                _run_recut(job)
            else:
                clip = next(c for c in job["clips"] if c["id"] == cid)
                words = []
                if clip.get("captions", True):
                    words = json.loads((Path(job["dir"]) / "words.json").read_text(encoding="utf-8"))
                render_one(job, clip, words)
        except Exception as e:  # noqa: BLE001
            (Path(job["dir"]) / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
            if kind in ("job", "recut"):
                for c in job["clips"]:
                    if c.get("status") in ("queued", "rendering"):
                        c.update(status="error", error="The video stopped before this clip was made.")
                _set(job, status="done" if kind == "recut" else "error", error=safe.friendly(e), stage="Stopped")
                creators.log(job["creator"], f"Video failed: {safe.friendly(e)[:200]}")
        finally:
            ctx._uid.reset(token)
            if kind in ("job", "recut"):
                _q.done()


def _pick_up_after_restart():
    """Videos and single clips that were being made when the server stopped: queued again, oldest first."""
    waiting = sorted((j for j in all_jobs() if j.get("status") == "queued" and j.get("resume")), key=lambda j: j["created"])
    for j in waiting:
        _q.put(("job", j["id"], None), user=j["user"])
    for j in all_jobs():
        for cid in j.pop("remake", []):
            _q.put(("clip", j["id"], cid))
        if j.get("status") == "done":
            save(j)


def start():
    load.start()
    load_all()
    _pick_up_after_restart()
    for _ in range(max(1, config.JOB_WORKERS)):
        threading.Thread(target=_worker, daemon=True).start()
    threading.Thread(target=_worker, args=(True,), daemon=True).start()   # re-making a single clip never waits behind whole videos
    threading.Thread(target=_live_worker, daemon=True).start()
    threading.Thread(target=_live_long_worker, daemon=True).start()
