"""Google Drive archive for finished clips.

Each user can connect one Google Drive account in Settings, Storage. PeakCut only asks for permission to manage the
files and folders it creates itself (drive.file), organises clips under PeakCut / <Creator> / <YYYY-MM>, uploads large
videos in 8 MB resumable pieces, checks that Google Drive recorded the exact byte size, and fetches archived clips back
into a short-lived local cache whenever someone plays, downloads or re-posts them."""
import json
import math
import os
import queue
import re
import secrets
import threading
import time
from pathlib import Path
from urllib.parse import urlencode

import httpx

from . import accounts, config, ctx

SCOPES = "https://www.googleapis.com/auth/drive.file https://www.googleapis.com/auth/userinfo.email"
CHUNK = 8 * 1024 * 1024
CACHE_TTL = 3600
_MAX_STATES = 5000

DEFAULTS = {
    "connected": False,
    "email": "",
    "refresh_token": "",
    "access_token": "",
    "expires": 0.0,
    "auto_after_post": True,
    "auto_days": 3,
    "folders": {},
    "archived_bytes": 0,
    "archived_count": 0,
    "last_error": "",
}

_lock = threading.RLock()
_mem: dict = {}
_states: dict = {}          # state -> (uid, created_at)
_inflight: set = set()      # (uid, jid, cid) items queued or uploading
_queue: "queue.Queue" = queue.Queue()
_retries: dict = {}         # (uid, jid, cid) -> (tries, due_at, manual)
_attempts: dict = {}        # (uid, jid, cid) -> consecutive failed attempts
_exhausted: set = set()     # (uid, jid, cid) items that failed 3 times until retried manually or reconnected
_started = False
_cache_guard = threading.Lock()
_cache_locks: dict = {}     # "{uid}_{did}" -> Lock


class FolderGone(RuntimeError):
    """Raised when a cached Google Drive folder was deleted or moved to Trash."""


def _file(create: bool = False) -> Path:
    d = config.user_dir(ctx.uid()) / "accounts"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d / "gdrive.json"


def _nonneg_int(val) -> int:
    if val is None or isinstance(val, bool):
        return 0
    try:
        return max(0, int(val))
    except (TypeError, ValueError):
        return 0


def _copy_cfg(data: dict) -> dict:
    src = data if isinstance(data, dict) else {}
    raw_exp = src.get("expires")
    try:
        expires = float(raw_exp) if raw_exp is not None and not isinstance(raw_exp, bool) else 0.0
        if not math.isfinite(expires):
            expires = 0.0
    except (TypeError, ValueError):
        expires = 0.0

    raw_days = src.get("auto_days", DEFAULTS["auto_days"])
    if raw_days is None or isinstance(raw_days, bool):
        auto_days = DEFAULTS["auto_days"]
    else:
        try:
            auto_days = max(0, min(365, int(raw_days)))
        except (TypeError, ValueError):
            auto_days = DEFAULTS["auto_days"]

    raw_folders = src.get("folders")
    folders = (
        {str(k): str(v) for k, v in raw_folders.items() if isinstance(k, str) and isinstance(v, str) and v}
        if isinstance(raw_folders, dict)
        else {}
    )
    raw_auto_post = src.get("auto_after_post")
    return {
        "connected": bool(src.get("connected", False)),
        "email": str(src.get("email") or ""),
        "refresh_token": str(src.get("refresh_token") or ""),
        "access_token": str(src.get("access_token") or ""),
        "expires": expires,
        "auto_after_post": DEFAULTS["auto_after_post"] if raw_auto_post is None else bool(raw_auto_post),
        "auto_days": auto_days,
        "folders": folders,
        "archived_bytes": _nonneg_int(src.get("archived_bytes")),
        "archived_count": _nonneg_int(src.get("archived_count")),
        "last_error": str(src.get("last_error") or ""),
    }


def _cfg() -> dict:
    u = ctx.uid()
    with _lock:
        if u not in _mem:
            if not config.user_dir(u).exists():
                return _copy_cfg({})
            f = _file(create=False)
            loaded = {}
            if f.is_file():
                try:
                    parsed = json.loads(f.read_text(encoding="utf-8"))
                    if isinstance(parsed, dict):
                        loaded = parsed
                except (OSError, ValueError):
                    loaded = {}
            _mem[u] = _copy_cfg(loaded)
        return _copy_cfg(_mem[u])


def _atomic_replace(src: Path, dst: Path) -> None:
    for attempt in range(6):
        try:
            src.replace(dst)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.02 * (attempt + 1))


def _save(cfg: dict) -> dict:
    u = ctx.uid()
    clean = _copy_cfg(cfg)
    with _lock:
        f = _file(create=True)
        tmp = f.with_name("gdrive.json.tmp")
        tmp.write_text(json.dumps(clean, indent=1), encoding="utf-8")
        try:
            tmp.chmod(0o600)
        except OSError:
            pass
        _atomic_replace(tmp, f)
        _mem[u] = _copy_cfg(clean)
        cfg.clear()
        cfg.update(_copy_cfg(clean))
    return cfg


def forget_mem(uid: str):
    """Drop the in-memory settings cache for a user."""
    with _lock:
        _mem.pop(uid, None)


def _purge_user_cache(uid: str):
    if not uid:
        return
    cdir = config.TMP_DIR / "drive_cache"
    if not cdir.is_dir():
        return
    prefix = f"{uid}_"
    for f in list(cdir.iterdir()):
        if not f.is_file() or not f.name.startswith(prefix):
            continue
        if f.name.endswith(".mp4"):
            lk = _key_lock(f.stem)
            acquired = lk.acquire(blocking=False)
            try:
                f.unlink(missing_ok=True)
            except OSError:
                pass
            finally:
                if acquired:
                    lk.release()
        else:
            try:
                f.unlink(missing_ok=True)
            except OSError:
                pass


def forget(uid: str):
    """Remove all in-memory state, pending OAuth tokens, queue entries and cached clips for a deleted user."""
    with _lock:
        with _queue.mutex:
            kept = [it for it in _queue.queue if it[0] != uid]
            removed_q = len(_queue.queue) - len(kept)
            if removed_q:
                _queue.queue.clear()
                _queue.queue.extend(kept)
                _queue.unfinished_tasks = max(0, _queue.unfinished_tasks - removed_q)
                if _queue.unfinished_tasks == 0:
                    _queue.all_tasks_done.notify_all()
        _mem.pop(uid, None)
        for k in [k for k, v in list(_states.items()) if v[0] == uid]:
            _states.pop(k, None)
        for item in [it for it in list(_inflight) if it[0] == uid]:
            _inflight.discard(item)
        for k in [k for k in list(_retries) if k[0] == uid]:
            _retries.pop(k, None)
        for k in [k for k in list(_attempts) if k[0] == uid]:
            _attempts.pop(k, None)
        for k in [k for k in list(_exhausted) if k[0] == uid]:
            _exhausted.discard(k)
        if uid:
            try:
                (config.user_dir(uid) / "accounts" / "gdrive.json").unlink(missing_ok=True)
            except OSError:
                pass
    _purge_user_cache(uid)


def _redact(text: str, cfg: dict | None = None) -> str:
    out = str(text or "")
    if not out:
        return ""
    secrets_to_hide = set()
    if isinstance(cfg, dict):
        for k in ("access_token", "refresh_token"):
            v = cfg.get(k)
            if isinstance(v, str) and v:
                secrets_to_hide.add(v)
    u = ctx.uid_or_none()
    if u:
        with _lock:
            mem_c = _mem.get(u)
            if isinstance(mem_c, dict):
                for k in ("access_token", "refresh_token"):
                    v = mem_c.get(k)
                    if isinstance(v, str) and v:
                        secrets_to_hide.add(v)
    try:
        _, g_secret = accounts.app_keys("google")
        if g_secret:
            secrets_to_hide.add(g_secret)
    except Exception:  # noqa: BLE001
        pass
    for s in sorted(secrets_to_hide, key=len, reverse=True):
        out = out.replace(s, "[redacted]")
    out = re.sub(r"(?i)\bBearer\s+[^\s\"',;]+", "Bearer [redacted]", out)
    out = re.sub(r"upload_id=[^\s&\"']+", "upload_id=[redacted]", out)
    out = re.sub(
        r'("?(?:access_token|refresh_token|client_secret)"?\s*[:=]\s*"?)[^"\s,&}]+',
        r"\1[redacted]",
        out,
        flags=re.IGNORECASE,
    )
    return out


def _check(r, what: str, cfg: dict | None = None):
    if r.status_code >= 400:
        raise RuntimeError(f"{what} failed ({r.status_code}): {_redact((r.text or '')[:300], cfg)}")
    return r


# ------------------------------------------------------------------ OAuth and settings
def can_connect() -> bool:
    return all(accounts.app_keys("google"))


def is_connected() -> bool:
    c = _cfg()
    return bool(c.get("connected") and c.get("refresh_token"))


def redirect_uri() -> str:
    return f"{config.PUBLIC_URL}/oauth/gdrive/callback"


def connect_url() -> str:
    if not can_connect():
        raise ValueError("Connecting Google Drive isn't set up on this server yet.")
    cid, _ = accounts.app_keys("google")
    now = time.time()
    state = secrets.token_urlsafe(24)
    with _lock:
        for k in [k for k, v in list(_states.items()) if now - v[1] > 900]:
            _states.pop(k, None)
        if len(_states) >= _MAX_STATES:
            excess = len(_states) - (_MAX_STATES - 1)
            for k, _ in sorted(_states.items(), key=lambda kv: kv[1][1])[:excess]:
                _states.pop(k, None)
        _states[state] = (ctx.uid(), now)
    return "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode({
        "client_id": cid,
        "redirect_uri": redirect_uri(),
        "response_type": "code",
        "scope": SCOPES,
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    })


def _parse_quota(sq) -> dict | None:
    if not isinstance(sq, dict):
        return None
    try:
        usage = int(sq.get("usage") or 0)
        limit = int(sq.get("limit") or 0)
    except (TypeError, ValueError):
        return None
    return {
        "limit_gb": round(limit / 1e9, 1) if limit else None,
        "used_gb": round(usage / 1e9, 1),
        "free_gb": round(max(0, limit - usage) / 1e9, 1) if limit else None,
    }


def connect_finish(code: str, state: str) -> dict:
    with _lock:
        got = _states.pop(state or "", None)
    if not got or got[0] != ctx.uid() or time.time() - got[1] > 900:
        raise ValueError("This sign-in has expired. Press Connect again.")
    code = (code or "").strip()
    if not code:
        raise ValueError("Google did not send a sign-in code. Try again.")
    cid, secret = accounts.app_keys("google")
    tok = _check(httpx.post("https://oauth2.googleapis.com/token", data={
        "code": code,
        "client_id": cid,
        "client_secret": secret,
        "redirect_uri": redirect_uri(),
        "grant_type": "authorization_code",
    }, timeout=30), "Google Drive sign-in").json()
    if not tok.get("refresh_token"):
        raise ValueError(
            "Google did not return a long-lived sign-in. Remove VICE CUT AI's access in your Google account and connect again."
        )
    if not tok.get("access_token"):
        raise RuntimeError("Google did not return an access token.")
    about = _check(httpx.get(
        "https://www.googleapis.com/drive/v3/about",
        params={"fields": "storageQuota,user"},
        headers={"Authorization": f"Bearer {tok['access_token']}"},
        timeout=30,
    ), "Reading Google Drive account", tok).json()
    email = str((about.get("user") or {}).get("emailAddress") or "").strip()
    with _lock:
        old = _cfg()
        same_email = bool(email and old.get("email") == email)
        cfg = {
            **old,
            "connected": True,
            "email": email,
            "refresh_token": str(tok["refresh_token"]),
            "access_token": str(tok["access_token"]),
            "expires": time.time() + float(tok.get("expires_in", 3600)) - 60,
            "folders": dict(old.get("folders") or {}) if same_email else {},
            "last_error": "",
        }
        _save(cfg)
        for k in [k for k in list(_exhausted) if k[0] == ctx.uid()]:
            _exhausted.discard(k)
    if not same_email:
        _purge_user_cache(ctx.uid())
    ctx.bump()
    st = public_status(include_quota=False)
    st["quota"] = _parse_quota(about.get("storageQuota"))
    return st


def disconnect() -> dict:
    u = ctx.uid()
    with _lock:
        cfg = _cfg()
        cfg.update(
            connected=False,
            email="",
            refresh_token="",
            access_token="",
            expires=0.0,
            folders={},
            last_error="",
        )
        _save(cfg)
        for k in [k for k in list(_retries) if k[0] == u]:
            _retries.pop(k, None)
        for k in [k for k in list(_attempts) if k[0] == u]:
            _attempts.pop(k, None)
        for k in [k for k in list(_exhausted) if k[0] == u]:
            _exhausted.discard(k)
    _purge_user_cache(u)
    ctx.bump()
    return public_status(include_quota=False)


def save_settings(patch: dict) -> dict:
    if not isinstance(patch, dict):
        raise ValueError("Invalid Google Drive settings.")
    with _lock:
        cfg = _cfg()
        if "auto_after_post" in patch:
            cfg["auto_after_post"] = bool(patch["auto_after_post"])
        if "auto_days" in patch:
            raw_days = patch["auto_days"]
            if isinstance(raw_days, bool):
                raise ValueError("Choose a number of days between 0 and 365.")
            try:
                days = int(raw_days)
            except (TypeError, ValueError) as e:
                raise ValueError("Choose a number of days between 0 and 365.") from e
            cfg["auto_days"] = max(0, min(365, days))
        _save(cfg)
    ctx.bump()
    return public_status(include_quota=False)


def _token(cfg: dict) -> str:
    if cfg.get("access_token") and cfg.get("expires", 0) > time.time():
        return cfg["access_token"]
    if not cfg.get("refresh_token"):
        raise RuntimeError("Google Drive is not connected.")
    cid, secret = accounts.app_keys("google")
    if not (cid and secret):
        raise RuntimeError("Google Drive sign-in is not set up on this server.")
    tok = _check(httpx.post("https://oauth2.googleapis.com/token", data={
        "client_id": cid,
        "client_secret": secret,
        "refresh_token": cfg["refresh_token"],
        "grant_type": "refresh_token",
    }, timeout=30), "Refreshing Google Drive sign-in", cfg).json()
    if not tok.get("access_token"):
        raise RuntimeError("Google Drive token refresh did not return an access token.")
    new_access = str(tok["access_token"])
    new_refresh = str(tok["refresh_token"]) if tok.get("refresh_token") else ""
    new_expires = time.time() + float(tok.get("expires_in", 3600)) - 60

    if ctx.uid_or_none():
        with _lock:
            live = _cfg()
            if not (live.get("connected") and live.get("refresh_token")):
                raise RuntimeError("Google Drive was disconnected.")
            live["access_token"] = new_access
            if new_refresh:
                live["refresh_token"] = new_refresh
            live["expires"] = new_expires
            _save(live)
            cfg.clear()
            cfg.update(_copy_cfg(live))
        return cfg["access_token"]

    if not (cfg.get("connected") and cfg.get("refresh_token")):
        raise RuntimeError("Google Drive was disconnected.")
    cfg["access_token"] = new_access
    if new_refresh:
        cfg["refresh_token"] = new_refresh
    cfg["expires"] = new_expires
    return cfg["access_token"]


def quota(cfg: dict = None) -> dict | None:
    c = cfg if cfg is not None else _cfg()
    if not (c.get("connected") and c.get("refresh_token")):
        return None
    try:
        token = _token(c)
        r = httpx.get(
            "https://www.googleapis.com/drive/v3/about",
            params={"fields": "storageQuota,user"},
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
        )
        if r.status_code != 200:
            return None
        return _parse_quota(r.json().get("storageQuota"))
    except Exception:  # noqa: BLE001
        return None


def queued_for_me() -> int:
    u = ctx.uid_or_none()
    if not u:
        return 0
    with _lock:
        return sum(1 for item in _inflight if item[0] == u)


def public_status(include_quota: bool = True) -> dict:
    cfg = _cfg()
    conn = bool(cfg.get("connected") and cfg.get("refresh_token"))
    return {
        "ready": can_connect(),
        "redirect": redirect_uri(),
        "connected": conn,
        "email": cfg.get("email", "") if conn else "",
        "auto_after_post": bool(cfg.get("auto_after_post", True)),
        "auto_days": int(cfg.get("auto_days", 3)),
        "archived_bytes": int(cfg.get("archived_bytes", 0)),
        "archived_count": int(cfg.get("archived_count", 0)),
        "last_error": _redact(str(cfg.get("last_error") or ""), cfg),
        "quota": quota(cfg) if (include_quota and conn) else None,
        "queued": queued_for_me(),
    }


# ------------------------------------------------------------------ Folder tree
def drop_folder_cache(cfg: dict):
    """Clear cached Drive folder IDs when a folder was deleted or moved."""
    cfg["folders"] = {}
    if ctx.uid_or_none():
        with _lock:
            live = _cfg()
            if live.get("connected") and live.get("email") == cfg.get("email"):
                live["folders"] = {}
                _save(live)
                cfg.clear()
                cfg.update(_copy_cfg(live))


def _esc_q(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _assert_folder_alive(token: str, folder_id: str, cfg: dict | None = None):
    r = httpx.get(
        f"https://www.googleapis.com/drive/v3/files/{folder_id}",
        params={"fields": "id,trashed"},
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    if r.status_code == 404:
        raise FolderGone("The Google Drive folder no longer exists.")
    meta = _check(r, "Checking Google Drive folder", cfg).json()
    if not isinstance(meta, dict) or not meta.get("id") or meta.get("trashed"):
        raise FolderGone("The Google Drive folder was moved to Trash.")


def _find_or_create_folder(token: str, name: str, parent: str | None, cfg: dict | None = None) -> str:
    if parent:
        _assert_folder_alive(token, parent, cfg)
    parent_ref = parent if parent else "root"
    q = (
        "mimeType='application/vnd.google-apps.folder' and trashed=false "
        f"and name='{_esc_q(name)}' and '{_esc_q(parent_ref)}' in parents"
    )
    auth_hdr = {"Authorization": f"Bearer {token}"}
    search = httpx.get(
        "https://www.googleapis.com/drive/v3/files",
        params={"q": q, "fields": "files(id,name)", "pageSize": 1},
        headers=auth_hdr,
        timeout=30,
    )
    if search.status_code == 404:
        raise FolderGone("Cached parent folder no longer exists on Google Drive.")
    files = _check(search, f"Searching Google Drive folder '{name}'", cfg).json().get("files") or []
    if files and files[0].get("id"):
        return str(files[0]["id"])

    body = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
    if parent:
        body["parents"] = [parent]
    create = httpx.post(
        "https://www.googleapis.com/drive/v3/files",
        params={"fields": "id,trashed"},
        json=body,
        headers=auth_hdr,
        timeout=30,
    )
    if create.status_code == 404:
        raise FolderGone("Cached parent folder no longer exists on Google Drive.")
    created = _check(create, f"Creating Google Drive folder '{name}'", cfg).json()
    if created.get("trashed"):
        raise FolderGone("Google Drive folder is inside Trash.")
    fid = created.get("id")
    if not fid:
        raise RuntimeError(f"Google Drive did not return an ID for folder '{name}'.")
    return str(fid)


def folder_for(cfg: dict, creator_name: str = "", ts: float = None) -> str:
    """Ensure PeakCut -> <creator> -> <YYYY-MM> exists on Google Drive and return the month folder ID."""
    creator = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "-", (creator_name or "").strip()).strip(" .")[:80] or "General"
    month = time.strftime("%Y-%m", time.localtime(ts or time.time()))
    c_key = f"c:{creator}"
    m_key = f"m:{creator}/{month}"
    folders = cfg.setdefault("folders", {})
    if folders.get(m_key):
        return folders[m_key]

    token = _token(cfg)
    folders = cfg.setdefault("folders", {})
    for attempt in range(2):
        try:
            changed = False
            made = {}
            root_id = folders.get("root")
            if not root_id:
                root_id = _find_or_create_folder(token, "PeakCut", None, cfg)
                folders["root"] = made["root"] = root_id
                changed = True
            creator_id = folders.get(c_key)
            if not creator_id:
                creator_id = _find_or_create_folder(token, creator, root_id, cfg)
                folders[c_key] = made[c_key] = creator_id
                changed = True
            month_id = folders.get(m_key)
            if not month_id:
                month_id = _find_or_create_folder(token, month, creator_id, cfg)
                folders[m_key] = made[m_key] = month_id
                changed = True
            if changed and ctx.uid_or_none():
                with _lock:
                    live = _cfg()
                    if live.get("connected") and live.get("email") == cfg.get("email"):
                        merged = dict(live.get("folders") or {})
                        merged.update(made)
                        live["folders"] = merged
                        _save(live)
                        cfg.clear()
                        cfg.update(_copy_cfg(live))
            return month_id
        except FolderGone:
            if attempt == 0:
                drop_folder_cache(cfg)
                folders = cfg.setdefault("folders", {})
                continue
            raise
    raise RuntimeError("Could not resolve Google Drive folder.")


_folder_for = folder_for


# ------------------------------------------------------------------ Resumable upload
def upload_mp4(cfg: dict, path: Path, folder_id: str, remote_name: str) -> dict:
    """Upload a local .mp4 in 8 MB resumable pieces and verify the remote byte count matches."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Clip file does not exist: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError("Clip file is empty.")

    token = _token(cfg)
    _assert_folder_alive(token, folder_id, cfg)
    name = (remote_name or path.name).strip()[:180] or path.name[:180]
    start = httpx.post(
        "https://www.googleapis.com/upload/drive/v3/files",
        params={"uploadType": "resumable", "fields": "id,size,webViewLink"},
        json={"name": name, "parents": [folder_id]},
        headers={
            "Authorization": f"Bearer {token}",
            "X-Upload-Content-Type": "video/mp4",
            "X-Upload-Content-Length": str(size),
        },
        timeout=60,
    )
    if start.status_code == 404:
        raise FolderGone("The target Google Drive folder no longer exists.")
    _check(start, "Starting Google Drive upload", cfg)
    upload_url = start.headers.get("Location") or start.headers.get("location")
    if not upload_url:
        raise RuntimeError("Google Drive did not return a resumable upload URL.")

    r = None
    with path.open("rb") as f:
        offset = 0
        while offset < size:
            part = f.read(CHUNK)
            if not part:
                raise RuntimeError("Clip file ended unexpectedly during upload.")
            first = offset
            last = offset + len(part) - 1
            r = httpx.put(
                upload_url,
                content=part,
                headers={
                    "Content-Length": str(len(part)),
                    "Content-Range": f"bytes {first}-{last}/{size}",
                },
                timeout=900,
            )
            offset = last + 1
            if offset < size:
                if r.status_code != 308:
                    detail = _redact((r.text or "")[:300], cfg)
                    raise RuntimeError(f"Google Drive chunk upload failed ({r.status_code}): {detail}")
            else:
                if r.status_code not in (200, 201):
                    detail = _redact((r.text or "")[:300], cfg)
                    raise RuntimeError(f"Google Drive upload failed ({r.status_code}): {detail}")

    try:
        done = r.json() if r is not None else {}
    except ValueError as e:
        raise RuntimeError("Google Drive returned an invalid JSON response.") from e
    if not isinstance(done, dict) or not done.get("id"):
        raise RuntimeError("Google Drive upload did not return a file ID.")
    try:
        remote_size = int(done.get("size") or -1)
    except (TypeError, ValueError):
        remote_size = -1
    if remote_size != size:
        raise RuntimeError("Google Drive upload size mismatch")

    fid = str(done["id"])
    return {
        "id": fid,
        "url": done.get("webViewLink") or f"https://drive.google.com/file/d/{fid}/view",
        "size": size,
        "at": time.time(),
    }


upload_file = upload_mp4


# ------------------------------------------------------------------ On-demand local cache
def _cache_dir() -> Path:
    d = config.TMP_DIR / "drive_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _key_lock(key: str) -> threading.Lock:
    with _cache_guard:
        lk = _cache_locks.get(key)
        if lk is None:
            if len(_cache_locks) > 512:
                for k in [k for k, v in list(_cache_locks.items()) if not v.locked()]:
                    _cache_locks.pop(k, None)
            lk = _cache_locks[key] = threading.Lock()
        return lk


def sweep_cache(max_age: int = CACHE_TTL) -> int:
    """Remove cached Google Drive clips from TMP_DIR/drive_cache older than max_age seconds."""
    cdir = config.TMP_DIR / "drive_cache"
    if not cdir.is_dir():
        return 0
    now = time.time()
    tmp_cutoff = max(max_age, 1800)
    removed = 0
    for f in list(cdir.iterdir()):
        if not f.is_file():
            continue
        name = f.name
        if name.endswith(".tmp"):
            try:
                if now - f.stat().st_mtime > tmp_cutoff:
                    f.unlink(missing_ok=True)
                    removed += 1
            except OSError:
                pass
            continue
        if name.endswith(".mp4"):
            lk = _key_lock(f.stem)
            if not lk.acquire(blocking=False):
                continue
            try:
                if f.is_file() and now - f.stat().st_mtime > max_age:
                    f.unlink(missing_ok=True)
                    removed += 1
            except OSError:
                pass
            finally:
                lk.release()
            continue
        try:
            if now - f.stat().st_mtime > max_age:
                f.unlink(missing_ok=True)
                removed += 1
        except OSError:
            pass
    return removed


def ensure_local(job: dict, clip: dict) -> Path | None:
    """Return the clip's local .mp4 if still on SSD, otherwise stream it from Google Drive into drive_cache."""
    if not isinstance(job, dict) or not isinstance(clip, dict):
        return None
    if clip.get("file") and job.get("dir"):
        local = Path(job["dir"]) / "clips" / Path(str(clip["file"])).name
        if local.is_file():
            return local

    drive_meta = clip.get("drive") if isinstance(clip.get("drive"), dict) else {}
    did = drive_meta.get("id")
    if not did or not re.fullmatch(r"[A-Za-z0-9_\-]{4,120}", str(did)):
        return None
    did = str(did)
    try:
        expected_size = int(drive_meta.get("size") or 0)
    except (TypeError, ValueError):
        return None
    if expected_size < 0:
        return None

    uid = job.get("user") or ctx.uid_or_none()
    if not uid or not re.fullmatch(r"[A-Za-z0-9_]{6,40}", str(uid)):
        return None

    cdir = _cache_dir()
    key = f"{uid}_{did}"
    dest = cdir / f"{key}.mp4"
    with _key_lock(key):
        try:
            if dest.is_file():
                cur_size = dest.stat().st_size
                if cur_size > 0 and (expected_size == 0 or cur_size == expected_size):
                    os.utime(dest, None)
                    return dest
        except OSError:
            pass

        tmp = cdir / f"{key}.{threading.get_ident()}.tmp"
        try:
            with ctx.as_user(uid):
                cfg = _cfg()
                if not (cfg.get("connected") and cfg.get("refresh_token")):
                    return None
                token = _token(cfg)
            url = f"https://www.googleapis.com/drive/v3/files/{did}?alt=media"
            headers = {"Authorization": f"Bearer {token}"}
            wrote = 0
            header_size = None
            with httpx.stream("GET", url, headers=headers, timeout=600, follow_redirects=True) as r:
                if r.status_code != 200:
                    return None
                hdrs = getattr(r, "headers", None) or {}
                cl_raw = hdrs.get("Content-Length") or hdrs.get("content-length")
                if cl_raw is not None and str(cl_raw).strip() != "":
                    try:
                        header_size = int(str(cl_raw).strip())
                    except ValueError:
                        return None
                    if header_size <= 0 or (expected_size > 0 and header_size != expected_size):
                        return None
                with tmp.open("wb") as out:
                    for chunk in r.iter_bytes(CHUNK):
                        if chunk:
                            out.write(chunk)
                            wrote += len(chunk)

            if wrote <= 0:
                tmp.unlink(missing_ok=True)
                return None
            if expected_size > 0 and wrote != expected_size:
                tmp.unlink(missing_ok=True)
                return None
            if header_size is not None and wrote != header_size:
                tmp.unlink(missing_ok=True)
                return None
            if not tmp.is_file() or tmp.stat().st_size != wrote:
                tmp.unlink(missing_ok=True)
                return None
            _atomic_replace(tmp, dest)
            sweep_cache()
            return dest
        except Exception:  # noqa: BLE001
            return None
        finally:
            tmp.unlink(missing_ok=True)


# ------------------------------------------------------------------ Archive queue, retries, SSD cleanup & sweep
def enqueue_clip(uid: str, jid: str, cid: str, manual: bool = False) -> bool:
    if not (isinstance(uid, str) and uid and isinstance(jid, str) and jid and isinstance(cid, str) and cid):
        return False
    key = (uid, jid, cid)
    with _lock:
        if key in _inflight:
            return False
        _inflight.add(key)
        prev = _retries.pop(key, None)
        if prev is not None:
            _attempts[key] = prev[0]
        if manual:
            _exhausted.discard(key)
        _queue.put((uid, jid, cid, bool(manual)))
    ctx.bump(uid)
    return True


def on_clip_posted(uid: str, jid: str, cid: str) -> bool:
    if not (uid and jid and cid):
        return False
    key = (uid, jid, cid)
    with ctx.as_user(uid):
        with _lock:
            if key in _retries or key in _exhausted:
                return False
            cfg = _cfg()
        if cfg["connected"] and cfg["refresh_token"] and cfg["auto_after_post"]:
            return enqueue_clip(uid, jid, cid, manual=False)
    return False


def request_archive(jid: str, cid: str) -> bool:
    from . import jobs
    uid = ctx.uid()
    if not is_connected():
        raise ValueError("Connect Google Drive in Settings, Storage first.")
    job = jobs.get(jid)
    clip = next((c for c in (job or {}).get("clips", []) if c.get("id") == cid), None)
    if not job or not clip or clip.get("status") != "done":
        raise ValueError("That clip is not ready to archive.")
    if clip.get("drive"):
        raise ValueError("This clip is already archived on Google Drive.")
    fname = Path(str(clip.get("file") or "")).name
    if not fname or not (Path(job["dir"]) / "clips" / fname).is_file():
        raise ValueError("That clip is not ready to archive.")
    key = (uid, jid, cid)
    with _lock:
        if key not in _inflight:
            _retries.pop(key, None)
            _attempts.pop(key, None)
            _exhausted.discard(key)
    return enqueue_clip(uid, jid, cid, manual=True)


def clean_job_source(job: dict, now: float = None, auto_days: int = 0) -> bool:
    from . import jobs, storage
    if not isinstance(job, dict) or not job.get("dir"):
        return False
    with jobs._lock:
        live_job = jobs._jobs.get(job.get("id")) if job.get("id") else None
        target = live_job if isinstance(live_job, dict) else job
        if target.get("status") != "done":
            return False
        clips = target.get("clips") or []
        if any(c.get("status") in ("queued", "rendering") for c in clips):
            return False

        t_now = time.time() if now is None else float(now)
        valid = [c for c in clips if c.get("status") != "error"]
        eligible = (
            (bool(valid) and all(c.get("status") == "done" and c.get("drive") for c in valid))
            or (not valid and auto_days > 0 and (t_now - float(target.get("created") or 0)) >= auto_days * 86400)
        )
        if not eligible:
            return False

        sp1 = target.get("source_path") or ""
        src_obj = target.get("source") if isinstance(target.get("source"), dict) else None
        sp2 = (src_obj.get("path") if src_obj else "") or ""
        if not sp1 and not sp2:
            return False

        jdir_res = Path(target["dir"]).resolve()
        freed = False
        still_locked = set()
        seen = set()
        for raw_p in (sp1, sp2):
            if not raw_p:
                continue
            s = str(raw_p)
            if s in seen:
                continue
            seen.add(s)
            p = Path(s)
            try:
                inside = p.resolve().parent == jdir_res
            except OSError:
                inside = False
            if inside and p.is_file() and p.name not in ("job.json", "words.json", "loudness.json"):
                try:
                    sz = p.stat().st_size
                    p.unlink()
                    if sz > 0:
                        freed = True
                except OSError:
                    pass
                if p.exists():
                    still_locked.add(s)

        changed = False
        if sp1 and str(sp1) not in still_locked and target.get("source_path"):
            target["source_path"] = ""
            changed = True
        if sp2 and str(sp2) not in still_locked and isinstance(target.get("source"), dict) and target["source"].get("path"):
            target["source"]["path"] = ""
            changed = True
        if job is not target:
            job["source_path"] = target.get("source_path", "")
            if isinstance(job.get("source"), dict):
                job["source"]["path"] = (target.get("source") or {}).get("path", "")
        if changed or freed:
            jobs.save(target)
            storage._used["at"] = 0
        return freed


def archive_one(uid: str, jid: str, cid: str, manual: bool = False) -> str:
    from . import alerts, creators, jobs, posting, storage
    key = (uid, jid, cid)
    try:
        with _key_lock(f"arch:{uid}:{jid}:{cid}"), ctx.as_user(uid):
            if not config.user_dir(uid).exists():
                return "skipped"
            job = jobs.get(jid)
            if not job:
                with _lock:
                    _retries.pop(key, None)
                    _attempts.pop(key, None)
                return "skipped"
            clip = next((c for c in job.get("clips", []) if c.get("id") == cid), None)
            cfg = _cfg()
            if (
                not (cfg["connected"] and cfg["refresh_token"])
                or not clip
                or clip.get("status") != "done"
                or clip.get("drive")
            ):
                if not clip or clip.get("drive"):
                    with _lock:
                        _retries.pop(key, None)
                        _attempts.pop(key, None)
                return "skipped"

            if not manual and posting.has_open_posts(jid, cid):
                return "skipped"

            fname = Path(str(clip.get("file") or "")).name
            if not clip.get("file") or not fname or not job.get("dir"):
                return "skipped"
            local = Path(job["dir"]) / "clips" / fname
            if not local.is_file():
                return "skipped"

            ver_before = clip.get("version", 0)
            c_obj = creators.get(job.get("creator")) if job.get("creator") else None
            creator_name = (c_obj.get("name") if c_obj else "") or ""
            ts = clip.get("made_at") or job.get("created") or time.time()
            raw_label = str(clip.get("title") or clip.get("hook") or cid)
            clean_label = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "-", raw_label).strip(" ._")[:80] or cid
            remote_name = f"{clean_label}_{jid[:15]}_{cid}.mp4"

            try:
                fid = folder_for(cfg, creator_name, ts=ts)
                try:
                    info = upload_mp4(cfg, local, fid, remote_name)
                except FolderGone:
                    drop_folder_cache(cfg)
                    fid = folder_for(cfg, creator_name, ts=ts)
                    info = upload_mp4(cfg, local, fid, remote_name)
            except Exception as e:  # noqa: BLE001
                if not config.user_dir(uid).exists() or jobs.get(jid) is None:
                    return "skipped"
                err = _redact(str(e), cfg)[:260]
                notify_exhausted = False
                with _lock:
                    live = _cfg()
                    if live.get("connected") and live.get("refresh_token") and jobs.get(jid) is not None:
                        live["last_error"] = err
                        _save(live)
                        prev_tries = _retries.get(key, (_attempts.get(key, 0),))[0]
                        tries = prev_tries + 1
                        if tries < 3:
                            _attempts[key] = tries
                            _retries[key] = (tries, time.time() + 600, bool(manual))
                        else:
                            _retries.pop(key, None)
                            _attempts.pop(key, None)
                            _exhausted.add(key)
                            notify_exhausted = True
                if notify_exhausted:
                    alerts.notify(
                        f"Google Drive archive failed ({clip.get('title') or cid}): {err[:160]}",
                        kind="post",
                        link="/app/settings/storage",
                    )
                ctx.bump(uid)
                return "failed"

            with jobs._lock:
                if (
                    not config.user_dir(uid).exists()
                    or jobs.get(jid) is not job
                    or clip not in job.get("clips", [])
                    or clip.get("status") != "done"
                    or clip.get("version", 0) != ver_before
                ):
                    return "skipped"
                clip["drive"] = info
                jobs.save(job)

            try:
                local.unlink(missing_ok=True)
            except OSError:
                pass

            jdir = Path(job["dir"])
            src_raw = clip.get("src")
            src_in_use = bool(
                src_raw
                and any(
                    other is not clip
                    and other.get("status") in ("queued", "rendering")
                    and str(other.get("src") or "") == str(src_raw)
                    for other in job.get("clips", [])
                )
            )
            if src_raw and not src_in_use:
                p = Path(str(src_raw))
                try:
                    mine = (
                        p.resolve().parent in ((jdir / "clips").resolve(), jdir.resolve())
                        and p.name.startswith(f"{cid}.src")
                    )
                except OSError:
                    mine = False
                if mine:
                    try:
                        p.unlink(missing_ok=True)
                    except OSError:
                        pass
                    with jobs._lock:
                        if clip in job.get("clips", []):
                            clip.pop("src", None)
                            if job.get("kind") != "live":
                                clip.pop("src_offset", None)
                                clip.pop("src_end", None)
                            jobs.save(job)

            clips_dir = jdir / "clips"
            if clips_dir.is_dir():
                for extra in list(clips_dir.glob(f"{cid}.src*")):
                    if extra.is_file():
                        try:
                            extra.unlink(missing_ok=True)
                        except OSError:
                            pass

            with _lock:
                _retries.pop(key, None)
                _attempts.pop(key, None)
                _exhausted.discard(key)
                if config.user_dir(uid).exists() and jobs.get(jid) is job:
                    live = _cfg()
                    live["archived_count"] = _nonneg_int(live.get("archived_count")) + 1
                    live["archived_bytes"] = _nonneg_int(live.get("archived_bytes")) + _nonneg_int(info.get("size"))
                    live["last_error"] = ""
                    _save(live)

            creators.log(job.get("creator"), f"Archived to Google Drive: {clip.get('title') or clip.get('hook') or cid}")
            clean_job_source(job)
            storage._used["at"] = 0
            ctx.bump(uid)
            return "archived"
    finally:
        removed = False
        with _lock:
            if key in _inflight:
                _inflight.discard(key)
                removed = True
        if removed:
            ctx.bump(uid)


def sweep_user(uid: str, now: float = None, force: bool = False) -> dict:
    from . import jobs, posting, storage
    if not uid:
        return {"queued": 0, "cleaned": 0}
    with ctx.as_user(uid):
        cfg = _cfg()
        if not (cfg["connected"] and cfg["refresh_token"]):
            return {"queued": 0, "cleaned": 0}
        now_ts = float(now) if now is not None else time.time()
        days = int(cfg["auto_days"])
        queued = 0
        cleaned = 0
        unlinked_leftover = False

        with _lock:
            due_retries = [
                (jid, cid, manual)
                for (r_uid, jid, cid), (_tries, due_at, manual) in list(_retries.items())
                if r_uid == uid and due_at <= now_ts
            ]
        for jid, cid, manual in due_retries:
            if enqueue_clip(uid, jid, cid, manual=manual):
                queued += 1

        for j in jobs.list_jobs():
            jid = j.get("id") or ""
            jdir = Path(j["dir"]) if j.get("dir") else None
            for c in j.get("clips", []):
                cid = c.get("id") or ""
                if not cid or c.get("status") != "done":
                    continue
                if c.get("drive"):
                    if jdir and not posting.has_open_posts(jid, cid):
                        clips_dir = jdir / "clips"
                        fname_d = Path(str(c.get("file") or "")).name
                        if fname_d:
                            leftover_mp4 = clips_dir / fname_d
                            if leftover_mp4.is_file():
                                try:
                                    leftover_mp4.unlink()
                                    unlinked_leftover = True
                                except OSError:
                                    pass
                        if clips_dir.is_dir():
                            for extra in list(clips_dir.glob(f"{cid}.src*")):
                                if extra.is_file():
                                    try:
                                        extra.unlink()
                                        unlinked_leftover = True
                                    except OSError:
                                        pass
                    continue
                if not force:
                    with _lock:
                        if (uid, jid, cid) in _retries or (uid, jid, cid) in _exhausted:
                            continue
                else:
                    with _lock:
                        if (uid, jid, cid) not in _inflight:
                            _retries.pop((uid, jid, cid), None)
                            _attempts.pop((uid, jid, cid), None)
                            _exhausted.discard((uid, jid, cid))
                fname = Path(str(c.get("file") or "")).name
                if not jdir or not fname or not (jdir / "clips" / fname).is_file():
                    continue
                if posting.has_open_posts(jid, cid):
                    continue
                posts_map = c.get("posts") if isinstance(c.get("posts"), dict) else {}
                was_posted = any(isinstance(p, dict) and p.get("status") == "posted" for p in posts_map.values())
                age_ok = days > 0 and (now_ts - float(c.get("made_at") or j.get("created") or now_ts)) >= days * 86400
                if force or (cfg["auto_after_post"] and was_posted) or age_ok:
                    if enqueue_clip(uid, jid, cid, manual=force):
                        queued += 1
            if clean_job_source(j, now=now_ts, auto_days=days):
                cleaned += 1

        if unlinked_leftover:
            storage._used["at"] = 0
            ctx.bump(uid)

        return {"queued": queued, "cleaned": cleaned}


def sweep_all(now: float = None) -> dict:
    from . import auth
    sweep_cache()
    total_q = 0
    total_c = 0
    for uid in auth.user_ids():
        try:
            res = sweep_user(uid, now=now, force=False)
            total_q += res.get("queued", 0)
            total_c += res.get("cleaned", 0)
        except Exception:  # noqa: BLE001
            pass
    return {"queued": total_q, "cleaned": total_c}


def drain_queue(timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + max(0.0, float(timeout))
    while True:
        try:
            item = _queue.get_nowait()
        except queue.Empty:
            if _queue.unfinished_tasks == 0:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)
            continue
        try:
            uid, jid, cid, manual = item
            archive_one(uid, jid, cid, manual=manual)
        except Exception:  # noqa: BLE001
            pass
        finally:
            _queue.task_done()


def _worker_loop():
    while True:
        item = _queue.get()
        try:
            uid, jid, cid, manual = item
            archive_one(uid, jid, cid, manual=manual)
        except Exception:  # noqa: BLE001
            pass
        finally:
            _queue.task_done()


def _sweep_loop():
    time.sleep(15)
    while True:
        try:
            sweep_all()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(60)


def start():
    global _started
    with _lock:
        if _started:
            return
        _started = True
    threading.Thread(target=_worker_loop, name="gdrive-worker", daemon=True).start()
    threading.Thread(target=_sweep_loop, name="gdrive-sweep", daemon=True).start()
