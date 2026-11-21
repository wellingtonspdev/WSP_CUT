"""Hybrid magnetic 1280x720 JPEG thumbnail generator (local PIL composition + optional Gemini AI)."""
import base64
import io
import re
import shutil
import tempfile
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

from . import ai, config, usage
from .media import run

THUMB_W, THUMB_H = 1280, 720
AI_IMAGE_MODELS = ("gemini-3.1-flash-image-preview", "gemini-2.5-flash-image", "gemini-3-pro-image")

_ERROR_PRIORITY = {
    "billing_cap": 100,
    "auth": 95,
    "no_key": 90,
    "rate_limit": 80,
    "timeout": 70,
    "connection": 65,
    "server_error": 55,
    "model_unavailable": 45,
    "safety_blocked": 35,
    "api_error": 10,
}

_VARIANT_PROMPTS = (
    "Boost subject lighting, sharpness, and dramatic color pop.",
    "Use punchy viral YouTube contrast, warm rim lighting, and vivid saturation.",
    "Use cinematic documentary lighting, crisp shadows, and high-clarity focal emphasis.",
    "Use electric studio rim glow, bold dynamic composition, and crisp highlight pop.",
)

_VARIANT_THEMES = (
    {"slash": (845, 780), "line": (255, 230, 0), "colors": ("#FFE600", "#FFFFFF"), "outlines": ((255, 230, 0), (255, 42, 42)), "bars": ((255, 32, 32), (255, 230, 0))},
    {"slash": (790, 855), "line": (0, 224, 255), "colors": ("#FFFFFF", "#00E5FF"), "outlines": ((0, 224, 255), (255, 255, 255)), "bars": ((0, 200, 255), (255, 255, 255))},
    {"slash": (820, 745), "line": (255, 136, 0), "colors": ("#FF9F1C", "#FFE600"), "outlines": ((255, 136, 0), (255, 230, 0)), "bars": ((255, 94, 0), (255, 230, 0))},
    {"slash": (870, 790), "line": (158, 255, 0), "colors": ("#B8FF2C", "#FFFFFF"), "outlines": ((158, 255, 0), (255, 255, 255)), "bars": ((130, 230, 0), (255, 255, 255))},
)


def classify_api_error(status_code: int | None = None, body: str = "", exc: Exception | None = None, provider: str = "Gemini") -> dict:
    """Turn raw HTTP/network image API failures into categorized human-readable messages."""
    safe_body = ai.redact(body or "")
    low = (safe_body + " " + ai.redact(str(exc or ""))).lower()

    if exc is not None:
        if isinstance(exc, httpx.TimeoutException) or "timed out" in low or "timeout" in low:
            return {
                "category": "timeout",
                "title": f"{provider} demorou para responder",
                "message": f"A geração de imagem no {provider} excedeu o tempo limite. Geramos uma colagem local no lugar para você não ficar sem capa.",
                "action": "Aguarde alguns segundos e clique em \"Gerar nova imagem\" novamente, ou use o modo Local.",
                "status": status_code,
            }
        if isinstance(exc, (httpx.ConnectError, httpx.NetworkError, httpx.RemoteProtocolError, ConnectionError, OSError)) or any(
            tok in low for tok in ("network", "dns", "ssl", "connection", "unreachable", "name or service")
        ):
            return {
                "category": "connection",
                "title": f"Sem conexão com a API do {provider}",
                "message": f"Não foi possível conectar ao servidor do {provider} (falha de rede, DNS ou TLS). Aplicamos o modo Local automaticamente.",
                "action": "Verifique sua internet/VPN/firewall e tente novamente.",
                "status": status_code,
            }

    if status_code in (429, 403) and any(tok in low for tok in ("spending cap", "spend cap", "billing", "budget", "payment", "insufficient_quota", "exceeded your current quota")):
        return {
            "category": "billing_cap",
            "title": f"Limite de gastos/cota da conta {provider} atingido",
            "message": f"Sua conta do {provider} atingiu o teto mensal de gastos ou está sem créditos de imagem no momento. Por isso, aplicamos a colagem Local.",
            "action": "Revise seu limite em AI Studio (ai.studio/spend) ou troque a chave do Gemini em Configurações → IA.",
            "status": status_code,
        }

    if status_code == 429 or any(tok in low for tok in ("rate limit", "too many requests", "resource_exhausted", "quota")):
        return {
            "category": "rate_limit",
            "title": "Muitas requisições em pouco tempo",
            "message": f"O {provider} limitou temporariamente novas imagens por excesso de pedidos. Geramos uma capa Local agora.",
            "action": "Espere cerca de 30–60 segundos antes de pedir outra imagem por IA.",
            "status": status_code,
        }

    if status_code in (401, 403) or any(tok in low for tok in ("api key not valid", "invalid_api_key", "permission_denied", "unauthorized")):
        return {
            "category": "auth",
            "title": f"Chave do {provider} inválida ou sem permissão",
            "message": f"A chave configurada não foi aceita pelo serviço de imagem do {provider}. Aplicamos a capa Local.",
            "action": "Confira sua chave API em Configurações → IA (Principal ou Fallback).",
            "status": status_code,
        }

    if status_code == 404 or any(tok in low for tok in ("not found", "not supported", "deprecated")):
        return {
            "category": "model_unavailable",
            "title": "Modelo de imagem indisponível",
            "message": f"O modelo de thumbnail do {provider} não está disponível para esta chave ou região. Geramos uma versão Local.",
            "action": "Use o modo Local ou atualize a chave/região da conta Gemini.",
            "status": status_code,
        }

    if status_code is not None and 500 <= status_code <= 599:
        return {
            "category": "server_error",
            "title": f"Instabilidade temporária no {provider}",
            "message": f"Os servidores do {provider} retornaram erro interno (HTTP {status_code}). Geramos a thumbnail Local automaticamente.",
            "action": "Tente novamente em instantes.",
            "status": status_code,
        }

    if status_code == 200 or any(tok in low for tok in ("safety", "prohibited_content", "blocked")):
        return {
            "category": "safety_blocked",
            "title": "Imagem bloqueada pelo filtro da IA",
            "message": "A IA não devolveu a imagem (possível bloqueio de conteúdo/texto no prompt). Aplicamos a versão Local.",
            "action": "Tente ajustar o texto da thumbnail ou gere outra variação de frame.",
            "status": status_code,
        }

    return {
        "category": "api_error",
        "title": f"Não foi possível gerar com a IA do {provider}",
        "message": "A API retornou uma resposta inesperada. Geramos uma nova colagem Local para você continuar.",
        "action": "Tente novamente ou mantenha o modo Local.",
        "status": status_code,
    }


def _short_text(text: str) -> str:
    """Sanitize and trim thumbnail text to 2-4 uppercase words."""
    clean = re.sub(r"[\x00-\x1f\x7f]+", " ", str(text or "")).strip()
    words = [w for w in re.split(r"\s+", clean) if w]
    if not words:
        return ""
    return " ".join(words[:4]).upper()[:36]


def _load_font(size: int):
    for name in ("Anton-Regular.ttf", "Poppins-ExtraBold.ttf"):
        p = config.FONTS_DIR / name
        if p.is_file():
            try:
                return ImageFont.truetype(str(p), size)
            except OSError:
                continue
    return ImageFont.load_default()


def _gemini_thumb_cfg() -> tuple[str, str]:
    """Resolve Gemini base_url and API key from active, fallback, saved Gemini profile, or .env."""
    if ai.provider() == "gemini":
        cfg = ai.current()
        addr = (cfg.get("base_url") or ai.PROVIDERS["gemini"][1]).rstrip("/")
        return addr, (cfg.get("key") or "")
    fb = ai.fallback()
    if fb.get("provider") == "gemini" and fb.get("key"):
        addr = (fb.get("base_url") or ai.PROVIDERS["gemini"][1]).rstrip("/")
        return addr, fb["key"]
    prof = ai._cfg_for("gemini")
    if prof.get("key"):
        return (prof.get("address") or ai.PROVIDERS["gemini"][1]).rstrip("/"), prof["key"]
    env_key = config.env("PEAKCUT_AI_KEY") or ""
    if env_key.startswith("AIza"):
        return ai.PROVIDERS["gemini"][1].rstrip("/"), env_key
    return "", ""


def _try_ai_thumb(frame_bytes: bytes, text: str, variant: int = 0) -> tuple[bytes | None, str, dict | None]:
    """Attempt Gemini image generation/editing when a Gemini key is configured; return (bytes, model, error_dict)."""
    best_err: dict | None = None

    def _keep(err: dict):
        nonlocal best_err
        if best_err is None or _ERROR_PRIORITY.get(err.get("category", ""), 0) > _ERROR_PRIORITY.get(best_err.get("category", ""), 0):
            best_err = err

    try:
        addr, key = _gemini_thumb_cfg()
        if not key or not addr:
            return None, "", {
                "category": "no_key",
                "title": "Nenhuma chave Gemini para imagem configurada",
                "message": "A geração de thumbnail por IA usa a API de imagem do Gemini, que ainda não está configurada. Geramos uma nova colagem Local no lugar.",
                "action": "Adicione uma chave Gemini em Configurações → IA (Principal ou Fallback) ou selecione o modo Local.",
                "status": None,
            }
        short = _short_text(text)
        style_hint = _VARIANT_PROMPTS[variant % len(_VARIANT_PROMPTS)]
        prompt = (
            "Create a high-contrast, magnetic 16:9 YouTube gaming/creator thumbnail from this frame. "
            f"{style_hint}"
            + (f' Overlay bold impactful text: "{short}".' if short else "")
        )
        b64 = base64.b64encode(frame_bytes).decode("ascii")
        body = {
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mimeType": "image/jpeg", "data": b64}},
                ],
            }],
            "generationConfig": {"responseModalities": ["IMAGE"]},
        }
        for model in AI_IMAGE_MODELS:
            try:
                r = httpx.post(
                    f"{addr}/{model}:generateContent",
                    json=body,
                    headers={"x-goog-api-key": key},
                    timeout=18.0,
                )
                if r.status_code != 200:
                    _keep(classify_api_error(status_code=r.status_code, body=r.text))
                    continue
                data = r.json()
                parts = (((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or [])
                for part in parts:
                    inline = part.get("inlineData") or part.get("inline_data")
                    if inline and inline.get("data"):
                        raw = base64.b64decode(inline["data"])
                        with Image.open(io.BytesIO(raw)) as im:
                            im.verify()
                        return raw, model, None
                _keep(classify_api_error(status_code=200, body=r.text))
            except Exception as exc:  # noqa: BLE001
                _keep(classify_api_error(exc=exc))
                continue
    except Exception as exc:  # noqa: BLE001
        _keep(classify_api_error(exc=exc))
    return None, "", best_err or classify_api_error()


def _add_vignette(canvas: Image.Image) -> Image.Image:
    """Add subtle bottom and left dark gradients so text stays readable."""
    overlay = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    band_top = int(THUMB_H * 0.58)
    for y in range(band_top, THUMB_H):
        t = (y - band_top) / max(1, THUMB_H - band_top)
        alpha = int(175 * (t ** 1.35))
        draw.line([(0, y), (THUMB_W, y)], fill=(0, 0, 0, alpha))
    side_w = int(THUMB_W * 0.42)
    for x in range(side_w):
        t = 1.0 - (x / max(1, side_w))
        alpha = int(105 * (t ** 1.5))
        draw.line([(x, 0), (x, THUMB_H)], fill=(0, 0, 0, alpha))
    return Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")


def _paste_facecam(canvas: Image.Image, raw: Image.Image, facecam: list | None) -> bool:
    """Crop webcam from raw frame, enlarge slightly with bold border, and anchor over its own corner so it never duplicates."""
    if not facecam or len(facecam) != 4:
        return False
    try:
        fx, fy, fw, fh = (int(round(float(v))) for v in facecam)
    except (TypeError, ValueError):
        return False
    if fw <= 8 or fh <= 8:
        return False
    x1 = max(0, min(raw.width - 1, fx))
    y1 = max(0, min(raw.height - 1, fy))
    x2 = max(x1 + 1, min(raw.width, fx + fw))
    y2 = max(y1 + 1, min(raw.height, fy + fh))
    if (x2 - x1) <= 8 or (y2 - y1) <= 8:
        return False

    cam = raw.crop((x1, y1, x2, y2))
    cam = ImageEnhance.Contrast(cam).enhance(1.12)
    cam = ImageEnhance.Color(cam).enhance(1.15)
    sx, sy = THUMB_W / max(1, raw.width), THUMB_H / max(1, raw.height)
    orig_w, orig_h = int(round((x2 - x1) * sx)), int(round((y2 - y1) * sy))
    target_w = max(orig_w + 60, 340)
    target_h = max(orig_h + 36, min(320, int(round(cam.height * (target_w / max(1, cam.width))))))
    cam = ImageOps.fit(cam, (target_w, target_h), method=Image.Resampling.LANCZOS)

    border = 7
    outer = 4
    framed = Image.new("RGB", (target_w + (border + outer) * 2, target_h + (border + outer) * 2), (0, 0, 0))
    inner = Image.new("RGB", (target_w + border * 2, target_h + border * 2), (255, 230, 0))
    inner.paste(cam, (border, border))
    framed.paste(inner, (outer, outer))
    # Anchor over the same corner the webcam sits in, completely covering the smaller original
    cx, cy = (x1 + x2) / 2.0 / max(1, raw.width), (y1 + y2) / 2.0 / max(1, raw.height)
    px = (THUMB_W - framed.width - 18) if cx >= 0.5 else 18
    py = 18 if cy < 0.5 else (THUMB_H - framed.height - 64)
    canvas.paste(framed, (px, py))
    return cy >= 0.5 and cx >= 0.5


def _paste_collage_split(canvas: Image.Image, primary: Image.Image, secondary: Image.Image | None, variant: int = 0) -> None:
    """Compose a dynamic 2-panel YouTube collage on the right third with an angled high-contrast divider slash."""
    theme = _VARIANT_THEMES[variant % len(_VARIANT_THEMES)]
    if secondary is not None:
        sec = secondary.copy()
    else:
        # Zoom in on the action center of the primary frame for a dramatic detail split panel
        w, h = primary.size
        zw, zh = max(16, int(w * 0.58)), max(16, int(h * 0.58))
        x0, y0 = max(0, (w - zw) // 2), max(0, (h - zh) // 3)
        sec = primary.crop((x0, y0, min(w, x0 + zw), min(h, y0 + zh)))
    sec = ImageEnhance.Contrast(sec).enhance(1.26)
    sec = ImageEnhance.Color(sec).enhance(1.30)
    sec = ImageEnhance.Sharpness(sec).enhance(1.22)
    panel_w = 505
    panel = ImageOps.fit(sec, (panel_w, THUMB_H), method=Image.Resampling.LANCZOS)

    right_layer = Image.new("RGB", (THUMB_W, THUMB_H), (0, 0, 0))
    right_layer.paste(panel, (THUMB_W - panel_w, 0))

    top_x, bot_x = theme["slash"]
    mask = Image.new("L", (THUMB_W, THUMB_H), 0)
    ImageDraw.Draw(mask).polygon([(top_x, 0), (THUMB_W, 0), (THUMB_W, THUMB_H), (bot_x, THUMB_H)], fill=255)
    canvas.paste(right_layer, (0, 0), mask)

    draw = ImageDraw.Draw(canvas)
    draw.line([(top_x, 0), (bot_x, THUMB_H)], fill=(0, 0, 0), width=16)
    draw.line([(top_x, 0), (bot_x, THUMB_H)], fill=theme["line"], width=6)


def _pick_best_frames_ai(frames: list[Path]) -> tuple[Path, Path | None]:
    """Use the vision model (e.g. nvidia/nemotron-3.5-super-vl:free) to pick the most expressive hero frame."""
    valid = [f for f in frames if f.is_file() and f.stat().st_size > 200]
    if not valid:
        return frames[0], None
    if len(valid) == 1:
        return valid[0], None
    if not ai.ready():
        return valid[0], valid[-1]
    try:
        tile_w, tile_h = 320, 180
        strip = Image.new("RGB", (tile_w * len(valid), tile_h), (0, 0, 0))
        sdraw = ImageDraw.Draw(strip)
        for i, fp in enumerate(valid):
            with Image.open(fp) as im:
                tile = ImageOps.fit(im.convert("RGB"), (tile_w, tile_h), method=Image.Resampling.BILINEAR)
            strip.paste(tile, (i * tile_w, 0))
            sdraw.rectangle([i * tile_w + 6, 6, i * tile_w + 36, 36], fill=(0, 0, 0))
            sdraw.text((i * tile_w + 15, 10), str(i + 1), fill=(255, 230, 0))
        buf = io.BytesIO()
        strip.save(buf, format="JPEG", quality=82)
        prompt = (
            f"This contact sheet has {len(valid)} numbered video frames (1 to {len(valid)}). "
            "Which frame has the most dramatic action, clearest facial reaction, and least motion blur for a YouTube thumbnail? "
            "Reply with ONLY the single digit (1, 2, or 3)."
        )
        ans, sent, wrote = ai.see(prompt, buf.getvalue(), timeout=10.0)
        if sent or wrote:
            usage.record(sent, wrote, "Thumbnail: best frame")
        m = re.search(r"([1-9])", str(ans or ""))
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(valid):
                hero = valid[idx]
                others = [f for j, f in enumerate(valid) if j != idx]
                return hero, (others[-1] if others else None)
    except Exception:  # noqa: BLE001
        pass
    return valid[0], valid[-1]


def _draw_headline(canvas: Image.Image, text: str, has_facecam: bool, variant: int = 0) -> None:
    short = _short_text(text)
    if not short:
        return
    theme = _VARIANT_THEMES[variant % len(_VARIANT_THEMES)]
    draw = ImageDraw.Draw(canvas)
    max_w = (THUMB_W - 440) if has_facecam else (THUMB_W - 120)
    words = short.split()
    if len(words) >= 3 and len(short) > 14:
        mid = len(words) // 2
        lines = [" ".join(words[:mid]), " ".join(words[mid:])]
    else:
        lines = [short]

    font_size = 112
    font = _load_font(font_size)
    while font_size > 48:
        font = _load_font(font_size)
        widest = max(draw.textbbox((0, 0), ln, font=font, stroke_width=8)[2] for ln in lines)
        if widest <= max_w:
            break
        font_size -= 6

    line_heights = []
    for ln in lines:
        bb = draw.textbbox((0, 0), ln, font=font, stroke_width=8)
        line_heights.append(bb[3] - bb[1])
    total_h = sum(line_heights) + max(0, len(lines) - 1) * 14
    x = 54
    y = THUMB_H - total_h - 68
    colors = theme["colors"]
    outlines = theme["outlines"]
    bars = theme["bars"]
    for idx, ln in enumerate(lines):
        bb = draw.textbbox((x, y), ln, font=font, stroke_width=8)
        pad_x, pad_y = 16, 8
        draw.rounded_rectangle(
            [bb[0] - pad_x, bb[1] - pad_y, bb[2] + pad_x, bb[3] + pad_y],
            radius=12,
            fill=(0, 0, 0),
            outline=outlines[idx % len(outlines)],
            width=3,
        )
        draw.rectangle(
            [bb[0] - pad_x, bb[1] - pad_y, bb[0] - pad_x + 10, bb[3] + pad_y],
            fill=bars[idx % len(bars)],
        )
        draw.text((x + 5, y + 5), ln, font=font, fill="#000000", stroke_width=8, stroke_fill="#000000")
        draw.text(
            (x, y),
            ln,
            font=font,
            fill=colors[idx % len(colors)],
            stroke_width=8,
            stroke_fill="#000000",
        )
        y += line_heights[idx] + 18


def _compose_local(
    frame_path: Path,
    out_jpg: Path,
    text: str = "",
    facecam: list | None = None,
    secondary_path: Path | None = None,
    collage: bool = True,
    variant: int = 0,
) -> Path:
    with Image.open(frame_path) as raw_in:
        raw = raw_in.convert("RGB")
    sec_raw = None
    if secondary_path and Path(secondary_path).is_file() and Path(secondary_path) != Path(frame_path):
        try:
            with Image.open(secondary_path) as sec_in:
                sec_raw = sec_in.convert("RGB")
        except Exception:  # noqa: BLE001
            sec_raw = None
    canvas = ImageOps.fit(raw, (THUMB_W, THUMB_H), method=Image.Resampling.LANCZOS)
    canvas = ImageEnhance.Contrast(canvas).enhance(1.22)
    canvas = ImageEnhance.Color(canvas).enhance(1.26)
    canvas = ImageEnhance.Sharpness(canvas).enhance(1.18)
    if collage:
        _paste_collage_split(canvas, raw, sec_raw, variant=variant)
    canvas = _add_vignette(canvas)
    has_cam = _paste_facecam(canvas, raw, facecam)
    _draw_headline(canvas, text, has_cam, variant=variant)
    canvas.save(out_jpg, format="JPEG", quality=92)
    return out_jpg


def generate_with_report(
    src: Path,
    out_jpg: Path,
    peak_t: float = 1.2,
    text: str = "",
    facecam: list | None = None,
    mode: str = "local",
    collage: bool = True,
    variant: int = 0,
    alt_peak_t: float | None = None,
) -> tuple[Path, dict]:
    """Extract frames around peak_t, build a 1280x720 thumbnail, and return (out_jpg, report_dict)."""
    out_jpg = Path(out_jpg)
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    tbase = config.TMP_DIR if len(str(config.TMP_DIR)) < 120 else None
    if tbase:
        tbase.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="pkth", dir=str(tbase) if tbase else None))
    try:
        frame_path = work / "frame.jpg"
        t = max(0.0, float(peak_t or 1.2))
        run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", str(src), "-frames:v", "1", "-q:v", "2", str(frame_path)])
        if not frame_path.is_file():
            run(["ffmpeg", "-y", "-v", "error", "-ss", "0.00", "-i", str(src), "-frames:v", "1", "-q:v", "2", str(frame_path)])
        if not frame_path.is_file():
            raise RuntimeError("Não foi possível extrair um frame deste vídeo.")
        candidates = [frame_path]
        shift = 2.5 + (max(0, int(variant)) % 3) * 1.4
        alt_times = []
        if alt_peak_t is not None and abs(float(alt_peak_t) - t) >= 0.25:
            alt_times.append(max(0.0, float(alt_peak_t)))
        alt_times.extend([max(0.0, t - shift) if t >= shift + 0.1 else max(0.0, t * 0.35), t + shift])
        for idx, at in enumerate(alt_times, 1):
            if abs(at - t) < 0.25:
                continue
            ap = work / f"alt{idx}.jpg"
            try:
                run(["ffmpeg", "-y", "-v", "error", "-ss", f"{at:.2f}", "-i", str(src), "-frames:v", "1", "-q:v", "2", str(ap)])
                if ap.is_file() and ap.stat().st_size > 200:
                    candidates.append(ap)
            except Exception:  # noqa: BLE001
                continue
        hero_path = frame_path
        sec_path = candidates[1] if len(candidates) > 1 else None
        if mode == "ai" and frame_path.is_file():
            if len(candidates) > 1:
                try:
                    hero_path, sec_path = _pick_best_frames_ai(candidates)
                except Exception:  # noqa: BLE001
                    pass
            ai_bytes, model_used, ai_err = _try_ai_thumb(hero_path.read_bytes(), text, variant=variant)
            if ai_bytes:
                try:
                    with Image.open(io.BytesIO(ai_bytes)) as im:
                        fitted = ImageOps.fit(im.convert("RGB"), (THUMB_W, THUMB_H), method=Image.Resampling.LANCZOS)
                        fitted.save(out_jpg, format="JPEG", quality=90)
                    return out_jpg, {
                        "requested_mode": "ai",
                        "used_mode": "ai",
                        "fallback": False,
                        "model": model_used,
                        "variant": variant,
                        "frame_sec": round(t, 2),
                        "error": None,
                    }
                except Exception as exc:  # noqa: BLE001
                    ai_err = classify_api_error(exc=exc)
            _compose_local(hero_path, out_jpg, text=text, facecam=facecam, secondary_path=sec_path, collage=collage, variant=variant)
            return out_jpg, {
                "requested_mode": "ai",
                "used_mode": "local",
                "fallback": True,
                "model": "",
                "variant": variant,
                "frame_sec": round(t, 2),
                "error": ai_err,
            }
        _compose_local(hero_path, out_jpg, text=text, facecam=facecam, secondary_path=sec_path, collage=collage, variant=variant)
        return out_jpg, {
            "requested_mode": "local",
            "used_mode": "local",
            "fallback": False,
            "model": "",
            "variant": variant,
            "frame_sec": round(t, 2),
            "error": None,
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


def generate(
    src: Path,
    out_jpg: Path,
    peak_t: float = 1.2,
    text: str = "",
    facecam: list | None = None,
    mode: str = "local",
    collage: bool = True,
    variant: int = 0,
) -> Path:
    """Extract frames around peak_t and build a 1280x720 collage JPEG thumbnail (AI with local fallback)."""
    out_path, _ = generate_with_report(
        src, out_jpg, peak_t=peak_t, text=text, facecam=facecam, mode=mode, collage=collage, variant=variant
    )
    return out_path
