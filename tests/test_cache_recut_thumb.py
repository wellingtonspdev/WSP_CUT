"""Battery of 30 real tests covering thumbnails, per-click variations, categorized API errors, transcript/segment cache, and Re-cut."""
import atexit
import base64
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if shutil.disk_usage(tempfile.gettempdir()).free >= 4 * 1024**3:
    DATA_DIR = Path(tempfile.mkdtemp(prefix="pc-recut-test-"))
else:
    DATA_DIR = Path(tempfile.mkdtemp(prefix=".pc-recut-test-", dir=str(ROOT)))

TMP_DIR = DATA_DIR / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ["PEAKCUT_DATA"] = str(DATA_DIR)
os.environ["TEMP"] = str(TMP_DIR)
os.environ["TMP"] = str(TMP_DIR)
os.environ["DATABASE_URL"] = ""
os.environ["PEAKCUT_ACCOUNTS"] = "0"
os.environ["PEAKCUT_SKIP_STATIC"] = "0"
tempfile.tempdir = str(TMP_DIR)

from studio import ai, brain, config, ctx, db, gdrive, jobs, media, render, server, thumbs  # noqa: E402

db.init()
render.install_fonts()


def _cleanup():
    try:
        db.engine.dispose()
    except Exception:
        pass
    shutil.rmtree(DATA_DIR, ignore_errors=True)


atexit.register(_cleanup)

fails = []


def check(name, got, want=True):
    ok = want(got) if callable(want) else got == want
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"   got {got!r}, want {want!r}"))
    if not ok:
        fails.append(name)


def _make_multiscene_mp4(path: Path, seconds: int = 10):
    """Create a real H.264/AAC MP4 with visually distinct animated scenes every second."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", f"testsrc2=size=1280x720:rate=15:duration={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(path),
        ],
        check=True,
    )


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _words_range(n_sec: int):
    out = []
    for i in range(n_sec):
        t0 = float(i)
        txt = f"palavra{i}." if (i % 6 == 5) else f"palavra{i}"
        out.append({"w": txt, "s": t0, "e": round(t0 + 0.45, 2)})
    return out


class _FakeResp:
    def __init__(self, status_code: int, text: str = "", json_data: dict | None = None):
        self.status_code = status_code
        self.text = text
        self._json = json_data if json_data is not None else {}

    def json(self):
        return self._json


sample_mp4 = DATA_DIR / "sample_multi.mp4"
_make_multiscene_mp4(sample_mp4, seconds=10)
sample_mp4_2 = DATA_DIR / "sample_multi_2.mp4"
shutil.copyfile(sample_mp4, sample_mp4_2)

words_420 = _words_range(420)
db_420 = np.zeros(430, dtype=np.float32)

# ==============================================================================
# BLOCO A — Thumbnails, variacoes por clique e erros humanos categorizados
# ==============================================================================

# [T01] thumbs.generate local real (1280x720 JPEG, acentos PT-BR)
t01_out = DATA_DIR / "t01.jpg"
thumbs.generate(sample_mp4, t01_out, peak_t=3.0, text="ATÉ NÃO DAR MAIS?!", mode="local", collage=True)
with Image.open(t01_out) as im:
    check("[T01] thumbs.generate local outputs 1280x720 JPEG with PT-BR accents", (im.size, im.format), ((1280, 720), "JPEG"))

# [T02] thumbs.generate local real com facecam
t02_out = DATA_DIR / "t02_face.jpg"
thumbs.generate(sample_mp4, t02_out, peak_t=3.0, text="ATÉ NÃO DAR MAIS?!", facecam=[20, 20, 220, 160], mode="local", collage=True)
with Image.open(t02_out) as im:
    check("[T02] thumbs.generate with facecam creates distinct 1280x720 JPEG", (im.size, _sha(t02_out) != _sha(t01_out)), ((1280, 720), True))

# [T03] thumbs.generate_with_report gera 4 variantes visuais distintas (0..3)
t03_hashes = set()
for v in range(4):
    vp = DATA_DIR / f"t03_v{v}.jpg"
    _, rep = thumbs.generate_with_report(sample_mp4, vp, peak_t=3.0, text="TESTE VARIANTES!", mode="local", collage=True, variant=v)
    with Image.open(vp) as im:
        if im.size == (1280, 720) and rep.get("used_mode") == "local" and rep.get("variant") == v:
            t03_hashes.add(_sha(vp))
check("[T03] thumbs.generate_with_report produces 4 distinct 1280x720 variants", len(t03_hashes), 4)

UID = "u_local"
with ctx.as_user(UID):
    j = jobs.create({"type": "url", "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "name": "vid"}, {"max_clips": 1}, enqueue=False)
    jdir = Path(j["dir"])
    (jdir / "clips").mkdir(parents=True, exist_ok=True)
    cached_src = jdir / "clips" / "clip01.src.mp4"
    shutil.copyfile(sample_mp4, cached_src)
    (jdir / "words.json").write_text(json.dumps(words_420), encoding="utf-8")
    jobs._save_loudness(jdir, 0.0, db_420)
    j.update(
        status="done", duration=420.0, width=1280, height=720,
        clips=[{
            "id": "clip01", "start": 0.0, "end": 9.0, "score": 85, "hook": "FIRST", "title": "First clip",
            "description": "d", "hashtags": ["gta"], "status": "done", "version": 1,
            "file": "clip01.mp4", "cover": "clip01.jpg", "src": str(cached_src), "src_offset": 0.0, "src_end": 10.0,
        }],
    )
    jobs.save(j)

    # [T04] 3 cliques consecutivos em regenerate_thumb geram imagens, variantes e peak_t diferentes
    snap = []
    for step in range(3):
        res_j = jobs.regenerate_thumb(j["id"], "clip01", text="CLIPE EPICO!", mode="local", new_variant=True)
        c1 = res_j["clips"][0]
        snap.append((c1["thumb_variant"], c1["version"], c1["peak_t"], (jdir / "clips" / "clip01.jpg").read_bytes()))
    check(
        "[T04] 3 consecutive regenerate_thumb clicks advance variant/version/peak_t with distinct bytes",
        ([x[0] for x in snap], [x[1] for x in snap], len({x[2] for x in snap}), len({x[3] for x in snap})),
        ([1, 2, 3], [2, 3, 4], 3, 3),
    )

    # [T05] regenerate_thumb(peak_t=2.5, new_variant=False) respeita peak_t manual e mantem thumb_variant
    res_fixed = jobs.regenerate_thumb(j["id"], "clip01", text="FIXO!", mode="local", peak_t=2.5, new_variant=False)
    check(
        "[T05] regenerate_thumb with new_variant=False keeps variant and honors manual peak_t",
        (res_fixed["clips"][0]["thumb_variant"], res_fixed["clips"][0]["peak_t"]),
        (3, 2.5),
    )

    # [T06] Protecoes: rejeita clipe em rendering (RuntimeError) e sem fonte de midia (ValueError)
    j["clips"][0]["status"] = "rendering"
    got_rt = False
    try:
        jobs.regenerate_thumb(j["id"], "clip01")
    except RuntimeError:
        got_rt = True
    j["clips"][0]["status"] = "done"

    j_nosrc = jobs.create({"type": "file", "path": "", "name": "gone.mp4"}, {}, enqueue=False)
    j_nosrc["clips"] = [{"id": "c1", "start": 0.0, "end": 5.0, "status": "done", "version": 1}]
    jobs.save(j_nosrc)
    got_ve = False
    try:
        jobs.regenerate_thumb(j_nosrc["id"], "c1")
    except ValueError:
        got_ve = True
    check("[T06] regenerate_thumb guards rendering status (RuntimeError) and missing media (ValueError)", (got_rt, got_ve), (True, True))

# [T07] classify_api_error cobre todas as 10 categorias em PT-BR
cats = {
    "billing_cap": thumbs.classify_api_error(429, "Your project has exceeded its monthly spending cap"),
    "rate_limit": thumbs.classify_api_error(429, "Too many requests rate limit"),
    "auth": thumbs.classify_api_error(403, "API key not valid"),
    "model_unavailable": thumbs.classify_api_error(404, "model not found"),
    "timeout": thumbs.classify_api_error(exc=httpx.ReadTimeout("timed out")),
    "connection": thumbs.classify_api_error(exc=httpx.ConnectError("dns fail")),
    "server_error": thumbs.classify_api_error(503, "service unavailable"),
    "safety_blocked": thumbs.classify_api_error(200, "SAFETY blocked"),
    "api_error": thumbs.classify_api_error(400, "unexpected error"),
}
with patch.object(thumbs, "_gemini_thumb_cfg", return_value=("", "")):
    _, _, no_key_err = thumbs._try_ai_thumb(b"abc", "TESTE")
cats["no_key"] = no_key_err or {}
all_10_ok = all(
    cats[k].get("category") == k and cats[k].get("title") and cats[k].get("message") and cats[k].get("action")
    for k in ("billing_cap", "rate_limit", "auth", "model_unavailable", "timeout", "connection", "server_error", "safety_blocked", "api_error", "no_key")
)
check("[T07] classify_api_error covers all 10 categories with title/message/action in PT-BR", all_10_ok, True)

# [T08] Redacao de chave de API em erros HTTP
with patch.object(ai, "current", return_value={"provider": "gemini", "models": [], "base_url": "", "key": "AIzaSecretUnitKey999"}):
    red_err = thumbs.classify_api_error(403, "Invalid key AIzaSecretUnitKey999 on request")
    check("[T08] classify_api_error never leaks API keys in output", "AIzaSecretUnitKey999" not in json.dumps(red_err), True)

# [T09] Priorizacao entre erros de modelos: 404 primeiro + 429 spending cap segundo -> billing_cap
with patch.object(thumbs, "_gemini_thumb_cfg", return_value=("https://generativelanguage.googleapis.com/v1beta/models", "AIzaFake")), \
     patch("studio.thumbs.httpx.post", side_effect=[
         _FakeResp(404, "model not found"),
         _FakeResp(429, "Your project has exceeded its monthly spending cap."),
         _FakeResp(503, "server busy"),
     ]):
    _, _, prio_err = thumbs._try_ai_thumb(b"framebytes", "CAP")
check("[T09] _try_ai_thumb prioritizes billing_cap over 404 and 503", (prio_err or {}).get("category"), "billing_cap")

client = TestClient(server.app, base_url="http://127.0.0.1:8765")
HDR = {"Origin": "http://127.0.0.1:8765"}

# [T10] POST /thumb modo AI com 429 spending cap -> fallback local real 1280x720 + thumb_report.category=billing_cap
with patch.object(thumbs, "_gemini_thumb_cfg", return_value=("https://generativelanguage.googleapis.com/v1beta/models", "AIzaFake")), \
     patch.object(thumbs, "_pick_best_frames_ai", side_effect=lambda fs: (fs[0], fs[-1])), \
     patch("studio.thumbs.httpx.post", return_value=_FakeResp(429, "Your project has exceeded its monthly spending cap.")):
    r10 = client.post(f"/api/jobs/{j['id']}/clips/clip01/thumb", json={"thumb_text": "IA COTA!", "thumb_mode": "ai", "new_variant": True}, headers=HDR)
rep10 = r10.json().get("thumb_report") or {}
check("[T10] POST /thumb AI 429 spending cap falls back cleanly with billing_cap report",
      (r10.status_code, rep10.get("fallback"), rep10.get("used_mode"), (rep10.get("error") or {}).get("category")),
      (200, True, "local", "billing_cap"))

# [T11] POST /thumb modo AI com ConnectError e ReadTimeout -> fallback local + categorias connection e timeout
with patch.object(thumbs, "_gemini_thumb_cfg", return_value=("https://generativelanguage.googleapis.com/v1beta/models", "AIzaFake")), \
     patch.object(thumbs, "_pick_best_frames_ai", side_effect=lambda fs: (fs[0], fs[-1])):
    with patch("studio.thumbs.httpx.post", side_effect=httpx.ConnectError("connection refused")):
        r11a = client.post(f"/api/jobs/{j['id']}/clips/clip01/thumb", json={"thumb_text": "REDE!", "thumb_mode": "ai"}, headers=HDR)
    with patch("studio.thumbs.httpx.post", side_effect=httpx.ReadTimeout("timed out")):
        r11b = client.post(f"/api/jobs/{j['id']}/clips/clip01/thumb", json={"thumb_text": "TIMEOUT!", "thumb_mode": "ai"}, headers=HDR)
check(
    "[T11] POST /thumb reports ConnectError=connection and ReadTimeout=timeout",
    (((r11a.json().get("thumb_report") or {}).get("error") or {}).get("category"),
     ((r11b.json().get("thumb_report") or {}).get("error") or {}).get("category")),
    ("connection", "timeout"),
)

# [T12] POST /thumb modo AI com resposta base64 valida da IA -> used_mode=ai, fallback=False, error=None
ai_img_buf = io.BytesIO()
Image.new("RGB", (640, 360), (220, 40, 130)).save(ai_img_buf, format="JPEG")
b64_img = base64.b64encode(ai_img_buf.getvalue()).decode("ascii")
fake_ai_json = {"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/jpeg", "data": b64_img}}]}}]}
with patch.object(thumbs, "_gemini_thumb_cfg", return_value=("https://generativelanguage.googleapis.com/v1beta/models", "AIzaFake")), \
     patch.object(thumbs, "_pick_best_frames_ai", side_effect=lambda fs: (fs[0], fs[-1])), \
     patch("studio.thumbs.httpx.post", return_value=_FakeResp(200, "", fake_ai_json)):
    r12 = client.post(f"/api/jobs/{j['id']}/clips/clip01/thumb", json={"thumb_text": "IA OK!", "thumb_mode": "ai"}, headers=HDR)
rep12 = r12.json().get("thumb_report") or {}
with Image.open(jdir / "clips" / "clip01.jpg") as im12:
    check("[T12] POST /thumb AI success saves 1280x720 with used_mode=ai and fallback=False",
          (r12.status_code, rep12.get("used_mode"), rep12.get("fallback"), rep12.get("error"), im12.size),
          (200, "ai", False, None, (1280, 720)))

# ==============================================================================
# BLOCO B — Cache de transcricao, loudness, URL e reuso de segmentos
# ==============================================================================

# [T13] jobs._norm_url normaliza diferentes URLs do mesmo video YouTube e preserva Twitch
n_watch = jobs._norm_url("https://www.youtube.com/watch?v=2dWUut1e_XI&t=42s")
n_short_url = jobs._norm_url("https://youtu.be/2dWUut1e_XI?si=abc")
n_live = jobs._norm_url("https://www.youtube.com/live/2dWUut1e_XI")
n_shorts = jobs._norm_url("https://youtube.com/shorts/2dWUut1e_XI")
n_twitch = jobs._norm_url("https://www.twitch.tv/videos/123456789")
check("[T13] _norm_url normalizes YouTube formats and preserves Twitch",
      (len({n_watch, n_short_url, n_live, n_shorts}), n_watch, "twitch.tv" in n_twitch),
      (1, "yt:2dWUut1e_XI", True))

# [T14] _save_loudness e _load_loudness fatiamento e fallback seguro
ld_dir = DATA_DIR / "ld_test"
ld_dir.mkdir(parents=True, exist_ok=True)
jobs._save_loudness(ld_dir, 10.0, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
arr_slice = jobs._load_loudness(ld_dir, 12.0, 17.0)
arr_miss = jobs._load_loudness(DATA_DIR / "nonexistent_ld", 0.0, 5.0)
check("[T14] _save_loudness/_load_loudness slices range and falls back to zeros cleanly",
      (round(float(arr_slice[0]), 1), round(float(arr_slice[2]), 1), len(arr_miss), float(np.sum(arr_miss))),
      (3.0, 5.0, 6, 0.0))

# [T15] Cache automatico entre jobs via _find_cached_job + _run_job (pula download_audio e transcribe)
with ctx.as_user(UID):
    j_src_cache = jobs.create({"type": "url", "url": "https://www.youtube.com/watch?v=CacheVid999", "name": "cache999"}, {"max_clips": 1}, enqueue=False)
    Path(j_src_cache["dir"]).mkdir(parents=True, exist_ok=True)
    (Path(j_src_cache["dir"]) / "words.json").write_text(json.dumps(words_420), encoding="utf-8")
    jobs._save_loudness(Path(j_src_cache["dir"]), 0.0, db_420)
    j_src_cache.update(status="done", lang="pt", duration=420.0)
    jobs.save(j_src_cache)

    j_dup = jobs.create({"type": "url", "url": "https://youtu.be/CacheVid999?si=xyz", "name": "dup"}, {"max_clips": 1, "min_len": 5, "max_len": 20}, enqueue=False)
    def _fake_dl_section(u, od, stem, *a, **kw):
        Path(od).mkdir(parents=True, exist_ok=True)
        return shutil.copyfile(sample_mp4, Path(od) / f"{stem}.mp4")

    with patch("studio.fetch_model.ensure", return_value=None), \
         patch.object(media, "info", return_value={"kind": "video", "duration": 420.0, "is_live": False, "title": "Dup Test", "uploader": "Ch"}), \
         patch.object(media, "save_thumb", return_value=False), \
         patch.object(media, "download_audio", side_effect=AssertionError("must not download audio on cached URL")), \
         patch("studio.transcribe.transcribe", side_effect=AssertionError("must not run ASR on cached URL")), \
         patch.object(brain, "ask_ai", return_value=([{"start": 1.0, "end": 7.0, "score": 91, "hook": "CACHE HIT", "title": "Via cache", "description": "d", "hashtags": ["gta"], "reason": "ok"}], "mock-ai", 40)), \
         patch.object(brain, "_snap", return_value=(1.0, 7.0)), \
         patch.object(media, "download_section", side_effect=_fake_dl_section):
        jobs._run_job(j_dup)
    check("[T15] _run_job reuses URL transcript cache without download_audio or ASR",
          (j_dup.get("transcript_from"), j_dup.get("status"), len(j_dup.get("clips", [])), j_dup["clips"][0]["status"] if j_dup.get("clips") else None),
          ("cache", "done", 1, "done"))

    # [T16] Isolamento multi-usuario: outro usuario nunca enxerga cache de u_local
    j_other = {"id": "other_u2", "user": "u_other_user", "source": {"type": "url", "url": "https://www.youtube.com/watch?v=CacheVid999"}}
    check("[T16] _find_cached_job enforces per-user tenant isolation", jobs._find_cached_job(j_other, 0.0, 100.0), None)

# [T17] clean_options e request_render limites (2-15m em 16:9 vs 180s em 9:16 + long_min/max preservados)
c_4m = jobs.clean_options({"min_len": 210, "max_len": 270, "layout": "horizontal"})
c_clamp = jobs.clean_options({"min_len": 999, "max_len": 1500})
with ctx.as_user(UID):
    j_lim = jobs.create({"type": "file", "name": "lim.mp4"}, {}, enqueue=False)
    j_lim["clips"] = [
        {"id": "s1", "start": 0, "end": 20, "format": "9:16", "layout": "blur", "status": "done"},
        {"id": "h1", "start": 0, "end": 60, "format": "16:9", "layout": "horizontal", "status": "done"},
    ]
    jobs.save(j_lim)
    with patch.object(jobs._q, "put"):
        jobs.request_render(j_lim["id"], "h1", {"start": 0, "end": 300})
        rej_916 = False
        try:
            jobs.request_render(j_lim["id"], "s1", {"start": 0, "end": 181})
        except ValueError:
            rej_916 = True
check("[T17] length limits: 2-15m clean_options, long limits untouched, 9:16 rejects >180s, 16:9 accepts 300s",
      (c_4m["min_len"], c_clamp["min_len"], c_clamp["max_len"], jobs.LIMITS["long_min_len"], jobs.LIMITS["long_max_len"], rej_916, j_lim["clips"][1]["end"]),
      (210, 600, 900, (300, 1200), (600, 1800), True, 300.0))

# [T18] brain.find_clips: exclusao simetrica + prompt para >=120s
prompts_t18 = []
def _ai_t18(prompt, schema=None, purpose="AI call", timeout=None):
    prompts_t18.append(prompt)
    return [
        {"start": 10.0, "end": 40.0, "score": 99, "hook": "OLD", "title": "Old", "description": "d", "hashtags": ["g"], "reason": "r"},
        {"start": 100.0, "end": 340.0, "score": 90, "hook": "NEW 4M", "title": "New 4m", "description": "d2", "hashtags": ["g"], "reason": "r2"},
    ], "mock", 50
with patch.object(brain, "ask_ai", side_effect=_ai_t18):
    picked_18, _ = brain.find_clips(words_420, db_420, 420.0, title="T18", max_clips=4, min_len=210, max_len=270, output_lang="pt", exclude=[{"start": 8.0, "end": 45.0}])
check("[T18] brain.find_clips rejects excluded overlap and adapts prompt for >=120s cuts",
      (len(picked_18), picked_18[0]["start"] >= 95.0, "8-45s" in prompts_t18[0], "self-contained narrative segments" in prompts_t18[0]),
      (1, True, True, True))

# [T19] fallback_by_loudness e find_long_blocks respeitam exclude
db_peaks = np.zeros(700, dtype=np.float32)
db_peaks[50:80] = 10.0
db_peaks[300:330] = 8.0
words_700 = _words_range(700)
fb_picks = brain.fallback_by_loudness(db_peaks, words_700, 1, 40, 700.0, output_lang="pt", exclude=[{"start": 35.0, "end": 85.0}])
with patch.object(brain, "ask_ai", return_value=([
    {"start": 10.0, "end": 350.0, "score": 95, "title": "Block1", "description": "d", "hashtags": [], "thumb_text": "B1", "peak_t": 50.0, "reason": "r"},
    {"start": 360.0, "end": 680.0, "score": 90, "title": "Block2", "description": "d", "hashtags": [], "thumb_text": "B2", "peak_t": 400.0, "reason": "r"},
], "mock", 30)):
    lb_picks, _ = brain.find_long_blocks(words_700, db_peaks, 700.0, max_blocks=1, min_len=300, max_len=360, exclude=[{"start": 15.0, "end": 345.0}])
check("[T19] fallback_by_loudness and find_long_blocks skip excluded ranges",
      (bool(fb_picks) and fb_picks[0]["start"] > 100.0, len(lb_picks) == 1 and lb_picks[0]["start"] >= 355.0),
      (True, True))

# [T20] _run_recut real em 16:9 (horizontal) reaproveitando .src.mp4 -> MP4 1920x1080 + thumb 1280x720
with ctx.as_user(UID):
    jobs.request_recut(j["id"], {"max_clips": 1, "min_len": 5, "max_len": 10, "layout": "horizontal", "thumb_mode": "local", "range_start": 0, "range_end": 120})
    with jobs._q.cv:
        while jobs._q.lines.get(UID):
            jobs._q.lines[UID].popleft()
    j["pending_recut"]["min_len"] = 3
    j["pending_recut"]["max_len"] = 9
    with patch.object(brain, "ask_ai", return_value=([{"start": 4.0, "end": 8.0, "score": 92, "hook": "NOVO", "title": "Recut 16:9", "description": "d", "hashtags": ["gta"], "thumb_text": "ABSURDO!", "reason": "ok"}], "mock", 25)), \
         patch.object(brain, "_snap", return_value=(4.0, 8.0)), \
         patch.object(media, "download_audio", side_effect=AssertionError("must not download audio on recut")), \
         patch.object(media, "download_section", side_effect=AssertionError("must reuse cached .src.mp4")):
        jobs._run_recut(j)
    c2 = next(c for c in j["clips"] if c["id"] == "clip02")
    p2_info = media.probe(Path(j["dir"]) / "clips" / c2["file"])
    with Image.open(Path(j["dir"]) / "clips" / c2["cover"]) as im20:
        check("[T20] real _run_recut renders 1920x1080 16:9 MP4 + 1280x720 JPEG from cached src",
              (c2["format"], c2["layout"], p2_info["width"], p2_info["height"], im20.size),
              ("16:9", "horizontal", 1920, 1080, (1280, 720)))

    # [T21] _run_recut filtra exclusao apenas por mesmo formato e respeita avoid_overlap=False
    captured_excl = []
    def _recut_spy(words_arg, *a, **kw):
        captured_excl.append(list(kw.get("exclude") or ()))
        return [{"start": 1.0, "end": 5.0, "score": 88, "hook": "H3", "title": "T3", "description": "d", "hashtags": [], "reason": "ok"}], {"model": "m", "tokens": 10, "note": ""}

    # clip01 is 9:16 [0..9], clip02 is 16:9 [4..8]; recutting as 9:16 with avoid_overlap=True excludes only clip01
    jobs.request_recut(j["id"], {"max_clips": 1, "min_len": 5, "max_len": 10, "layout": "blur", "avoid_overlap": True})
    with jobs._q.cv:
        while jobs._q.lines.get(UID):
            jobs._q.lines[UID].popleft()
    with patch.object(brain, "find_clips", side_effect=_recut_spy):
        jobs._run_recut(j)
    # Now with avoid_overlap=False, exclude must be empty
    jobs.request_recut(j["id"], {"max_clips": 1, "min_len": 5, "max_len": 10, "layout": "blur", "avoid_overlap": False})
    with jobs._q.cv:
        while jobs._q.lines.get(UID):
            jobs._q.lines[UID].popleft()
    with patch.object(brain, "find_clips", side_effect=_recut_spy):
        jobs._run_recut(j)
    c_ids = [c["id"] for c in j["clips"]]
    check("[T21] _run_recut filters exclude by same format, respects avoid_overlap=False, increments IDs",
          (len(captured_excl[0]), round(captured_excl[0][0]["start"], 1), len(captured_excl[1]), c_ids[:4]),
          (1, 0.0, 0, ["clip01", "clip02", "clip03", "clip04"]))

    # [T22] render_compilation + _next_id(comp) com multiplos segs em cache
    comp_mp4 = render.render_compilation(
        sample_mp4, Path(j["dir"]) / "clips", "comp_real",
        {"segments": [
            {"start": 0.5, "end": 2.5, "src": str(sample_mp4), "src_offset": 0.0},
            {"start": 4.0, "end": 6.0, "src": str(sample_mp4_2), "src_offset": 0.0},
        ], "thumb_text": "COMPILADO REAL!", "peak_t": 1.5},
        1280, 720,
    )
    comp_probe = media.probe(Path(j["dir"]) / "clips" / comp_mp4["file"])
    j["clips"].append({"id": "comp01", "start": 0, "end": 10, "status": "done"})
    next_comp = jobs._next_id(j, "comp")
    check("[T22] render_compilation uses multi-segment cached srcs to render 1920x1080 + _next_id -> comp02",
          (comp_probe["width"], comp_probe["height"], comp_probe["duration"] >= 3.5, next_comp),
          (1920, 1080, True, "comp02"))

# ==============================================================================
# BLOCO C — Rotas HTTP, restart, cookies opt-in, UI Liquid Glass e midia real
# ==============================================================================

# [T23] request_recut limpa range vazio e valida range_end <= range_start
with ctx.as_user(UID):
    jobs.request_recut(j["id"], {"range_start": 60, "range_end": 180})
    with jobs._q.cv:
        while jobs._q.lines.get(UID):
            jobs._q.lines[UID].popleft()
    j["status"] = "done"
    jobs.request_recut(j["id"], {"range_start": None, "range_end": ""})
    cleared_ok = j["pending_recut"]["range_start"] is None and j["pending_recut"]["range_end"] is None
    with jobs._q.cv:
        while jobs._q.lines.get(UID):
            jobs._q.lines[UID].popleft()
    j["status"] = "done"
    bad_range = False
    try:
        jobs.request_recut(j["id"], {"range_start": 200, "range_end": 100})
    except ValueError:
        bad_range = True
    check("[T23] request_recut clears empty ranges and rejects inverted range", (cleared_ok, bad_range), (True, True))

# [T24] Protecoes HTTP em POST /api/jobs/{jid}/recut: 400 / 409 / 404 / 200
with ctx.as_user(UID):
    j_nocache = jobs.create({"type": "url", "url": "https://www.youtube.com/watch?v=NoCache111", "name": "nc"}, {}, enqueue=False)
    j_nocache["status"] = "done"
    jobs.save(j_nocache)
    j_busy = jobs.create({"type": "url", "url": "https://www.youtube.com/watch?v=Busy111", "name": "busy"}, {}, enqueue=False)
    j_busy["status"] = "running"
    jobs.save(j_busy)

r24_400 = client.post(f"/api/jobs/{j_nocache['id']}/recut", json={}, headers=HDR)
r24_409 = client.post(f"/api/jobs/{j_busy['id']}/recut", json={}, headers=HDR)
r24_404 = client.post("/api/jobs/nonexistent_id_999/recut", json={}, headers=HDR)
with patch.object(jobs._q, "put") as mock_put24:
    r24_200 = client.post(f"/api/jobs/{j['id']}/recut", json={"max_clips": 2, "min_len": 210, "max_len": 270, "layout": "horizontal"}, headers=HDR)
    # Restore j status to done after HTTP enqueue check
    with ctx.as_user(UID):
        j["status"] = "done"
        j.pop("pending_recut", None)
        jobs.save(j)
check("[T24] POST /recut HTTP guards return 400 (no cache), 409 (running), 404 (missing), 200 (valid)",
      (r24_400.status_code, r24_409.status_code, r24_404.status_code, r24_200.status_code, mock_put24.called),
      (400, 409, 404, 200, True))

# [T25] Recuperacao apos restart (_load_file) preserva clipes feitos em job com pending_recut
rst_dir = DATA_DIR / "restart_job_dir"
rst_dir.mkdir(parents=True, exist_ok=True)
(rst_dir / "words.json").write_text("[]", encoding="utf-8")
rst_job = {
    "id": "rst_recut_01", "kind": "video", "created": 1.0,
    "source": {"type": "url", "url": "https://youtu.be/abc", "name": "r"},
    "options": {"captions": True, "output_lang": "pt"},
    "status": "running", "pending_recut": {"max_clips": 3},
    "clips": [
        {"id": "clip01", "start": 1, "end": 10, "status": "done"},
        {"id": "clip02", "start": 20, "end": 30, "status": "queued"},
    ],
}
(rst_dir / "job.json").write_text(json.dumps(rst_job), encoding="utf-8")
jobs._load_file(rst_dir / "job.json", UID)
loaded_rst = jobs._jobs["rst_recut_01"]
check("[T25] _load_file recovers interrupted pending_recut cleanly without losing done clips",
      (loaded_rst["status"], "pending_recut" in loaded_rst, loaded_rst["clips"][0]["status"], loaded_rst["clips"][1]["status"]),
      ("done", False, "done", "error"))

# [T26] media.yt_extra_opts opt-in limpo
with patch.object(config, "env", side_effect=lambda k, d="": "firefox" if k == "PEAKCUT_YT_COOKIES_BROWSER" else ""):
    opt_ff = media.yt_extra_opts()
cfile_tmp = DATA_DIR / "cookies.txt"
cfile_tmp.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
with patch.object(config, "env", side_effect=lambda k, d="": str(cfile_tmp) if k == "PEAKCUT_YT_COOKIES" else "firefox"):
    opt_cf = media.yt_extra_opts()
with patch.object(config, "env", return_value=""):
    opt_empty = media.yt_extra_opts()
check("[T26] yt_extra_opts honors firefox browser, prioritizes cookiefile, stays empty when unset",
      (opt_ff.get("cookiesfrombrowser"), opt_cf.get("cookiefile"), "cookiesfrombrowser" in opt_empty, "cookiefile" in opt_empty),
      (("firefox",), str(cfile_tmp), False, False))

# [T27] GET /api/jobs e /api/jobs/{jid} expoem has_cache e ocultam campos privados
r27_all = client.get("/api/jobs").json()
r27_one = client.get(f"/api/jobs/{j['id']}").json()
r27_nc = client.get(f"/api/jobs/{j_nocache['id']}").json()
priv_hidden = all(k not in r27_one for k in ("dir", "user", "source_path", "recover", "pending_recut"))
check("[T27] /api/jobs exposes has_cache=True/False and hides internal server fields",
      (any(x["id"] == j["id"] and x.get("has_cache") is True for x in r27_all), r27_one.get("has_cache"), r27_nc.get("has_cache"), priv_hidden),
      (True, True, False, True))

# [T28] Auditoria real do HTML/CSS servido em GET /app (VICE CUT AI + Liquid Glass + nossas funcoes + zero box-shadow)
r28 = client.get("/app")
html = r28.text
req_tokens = (
    "VICE CUT AI",
    "renderOptionsGrid",
    "readOptionsGrid",
    "2 min (90",
    "4 min (210",
    "5 min (270",
    "Mais cortes",
    "renderThumbNotice",
    "classifyClientThumbError",
    "Gerar nova imagem",
)
no_light_alerts = "background:#FFF8EB" not in html and "background:#ECFDF3" not in html
no_light_select = 'id="${prefix}long_mode" style="width:100%;height:40px;border-radius:10px;border:1px solid var(--line);padding:0 10px;background:#fff"' not in html
check("[T28] GET /app serves VICE CUT AI Liquid Glass UI with all new controls and zero box-shadow",
      (r28.status_code, all(tok in html for tok in req_tokens), "box-shadow" not in html.lower(), no_light_alerts, no_light_select),
      (200, True, True, True, True))

# [T29] Validacao real do JS inline de /app via node --check
scripts = re.findall(r"<script[^>]*>(.*?)</script>", html, re.DOTALL)
js_file = DATA_DIR / "app_inline_check.js"
js_file.write_text("\n;\n".join(s for s in scripts if s.strip()), encoding="utf-8")
node_bin = shutil.which("node")
if node_bin:
    nr = subprocess.run([node_bin, "--check", str(js_file)], capture_output=True, text=True, encoding="utf-8")
    check("[T29] /app inline JavaScript passes node --check", (nr.returncode, nr.stderr.strip()), (0, ""))
else:
    check("[T29] /app inline JavaScript non-empty (node missing)", len(scripts) > 0, True)

# [T30] Teste visual/pixel real em midia multi-cena (usa video real salvo no host se existir, senao sample_mp4)
real_host_src = Path(r"D:\peakcut_data\users\u_local\jobs\20261119-210226-35ed332338\clips\clip01.src.mp4")
t30_src = real_host_src if real_host_src.is_file() else sample_mp4
t30_a = DATA_DIR / "t30_a.jpg"
t30_b = DATA_DIR / "t30_b.jpg"
thumbs.generate_with_report(t30_src, t30_a, peak_t=1.2, text="CENA A!", mode="local", collage=True, variant=1, alt_peak_t=6.0)
thumbs.generate_with_report(t30_src, t30_b, peak_t=5.5, text="CENA A!", mode="local", collage=True, variant=2, alt_peak_t=2.0)
with Image.open(t30_a) as ia, Image.open(t30_b) as ib:
    arr_a = np.asarray(ia.convert("RGB"), dtype=np.float32)
    arr_b = np.asarray(ib.convert("RGB"), dtype=np.float32)
    mae = float(np.mean(np.abs(arr_a - arr_b)))
check("[T30] consecutive real-media thumbnails differ visually at pixel level (1280x720, MAE > 5)",
      (ia.size, ib.size, mae > 5.0),
      ((1280, 720), (1280, 720), True))

# [T31] Google Drive integration: clean_job_source preserves words.json + loudness.json; has_cache=False on cleaned local upload, True on URL
with ctx.as_user(UID):
    j_up_clean = jobs.create({"type": "file", "path": "", "name": "local_cleaned.mp4"}, {}, enqueue=False)
    udir = Path(j_up_clean["dir"])
    udir.mkdir(parents=True, exist_ok=True)
    raw_up = udir / "local_cleaned.mp4"
    shutil.copyfile(sample_mp4, raw_up)
    (udir / "words.json").write_text(json.dumps(words_420[:20]), encoding="utf-8")
    jobs._save_loudness(udir, 0.0, [1.0] * 15)
    j_up_clean.update(
        status="done", source_path=str(raw_up),
        clips=[{"id": "clip01", "start": 0, "end": 5, "status": "done", "drive": {"id": "drv1"}}],
    )
    j_up_clean["source"]["path"] = str(raw_up)
    jobs.save(j_up_clean)
    freed = gdrive.clean_job_source(j_up_clean)
check("[T31] gdrive.clean_job_source preserves words.json & loudness.json, and has_cache hides cleaned local upload",
      (freed, (udir / "words.json").is_file(), (udir / "loudness.json").is_file(), jobs.public(j_up_clean)["has_cache"], jobs.public(j)["has_cache"]),
      (True, True, True, False, True))

# [T32] regenerate_thumb on Drive-archived URL clip cleans temporary .src.mp4 from SSD after generating cover
with ctx.as_user(UID):
    j_drv = jobs.create({"type": "url", "url": "https://www.youtube.com/watch?v=DrvThumb999", "name": "drv"}, {}, enqueue=False)
    ddir = Path(j_drv["dir"])
    (ddir / "clips").mkdir(parents=True, exist_ok=True)
    j_drv.update(
        status="done", duration=60.0,
        clips=[{"id": "clip01", "start": 0.0, "end": 8.0, "status": "done", "version": 1, "drive": {"id": "drv_ok"}}],
    )
    jobs.save(j_drv)
    with patch.object(media, "download_section", side_effect=lambda u, od, stem, *a, **kw: shutil.copyfile(sample_mp4, Path(od) / f"{stem}.mp4")), \
         patch.object(media, "probe", return_value={"width": 1280, "height": 720, "duration": 10.0}):
        jobs.regenerate_thumb(j_drv["id"], "clip01", text="DRIVE THUMB!", mode="local")
    c_drv = j_drv["clips"][0]
    temp_cleaned = not (ddir / "clips" / "clip01.src.mp4").exists() and "src" not in c_drv
    cover_ok = (ddir / "clips" / "clip01.jpg").is_file()
check("[T32] regenerate_thumb on Drive-archived URL clip cleans temporary .src.mp4 after generating cover",
      (temp_cleaned, cover_ok),
      (True, True))

print(f"Summary: {32 - len(fails)}/32 real tests passed" if not fails else f"FAILED ({len(fails)}): {fails}")
if not fails:
    print("ALL PASSED")
sys.exit(1 if fails else 0)
