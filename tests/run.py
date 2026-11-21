"""Run the test suite against a throwaway copy of the app.

    python tests/run.py

Starts PeakCut on a spare port with an empty, temporary data folder, runs the tests against it, then stops it and
deletes the folder. Your real data and your running server are never touched. No AI key is needed: the tests cover accounts, permissions, limits and the queue, not clip quality."""
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORT = int(os.environ.get("PEAKCUT_TEST_PORT", "8799"))
BASE = f"http://127.0.0.1:{PORT}"


def _free_bytes(path: str | Path) -> int:
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return 0


def _rm_tree(path: Path) -> None:
    def _on_err(func, p, _exc):
        try:
            os.chmod(p, 0o700)
            func(p)
        except OSError:
            pass

    for _ in range(20):
        if not path.exists():
            return
        shutil.rmtree(path, onerror=_on_err)
        if not path.exists():
            return
        time.sleep(0.1)


def main() -> int:
    tmp_env = {}
    if _free_bytes(tempfile.gettempdir()) < 4 * 10**9 and _free_bytes(ROOT) >= 4 * 10**9:
        data = Path(tempfile.mkdtemp(prefix=".peakcut-test-", dir=str(ROOT)))
        sys_tmp = data / "sys_tmp"
        sys_tmp.mkdir(parents=True, exist_ok=True)
        tmp_env = {"TEMP": str(sys_tmp), "TMP": str(sys_tmp), "TMPDIR": str(sys_tmp)}
    else:
        data = Path(tempfile.mkdtemp(prefix="peakcut-test-"))
    env = {**os.environ, **tmp_env, "PEAKCUT_DATA": str(data), "PEAKCUT_PORT": str(PORT), "PEAKCUT_PUBLIC_URL": BASE,
           "PEAKCUT_HOST": "127.0.0.1", "PEAKCUT_AUTO_RELOAD": "0", "DATABASE_URL": "",
           "PEAKCUT_ACCOUNTS": "1", "PEAKCUT_FETCH_MODEL": "0", "PEAKCUT_UPDATE_CHECK": "0",                   # these tests are about people signing in; tests/test_local.py covers the default
           # nothing outside this machine is contacted: no AI, no email, no Google sign-in
           "PEAKCUT_SMTP_HOST": "", "PEAKCUT_VERIFY_EMAIL": "1",   # the tests cover the confirm-your-email rule
           "GOOGLE_CLIENT_ID": "", "GOOGLE_CLIENT_SECRET": "", "GOOGLE_CLIENT_FILE": "", "PYTHONUNBUFFERED": "1",
           "PEAKCUT_AI_KEY": "", "PEAKCUT_AI_KEY_FILE": "", "PEAKCUT_AI_MODELS": "",
           "PEAKCUT_AI_FALLBACK_PROVIDER": "", "PEAKCUT_AI_FALLBACK_MODELS": "", "PEAKCUT_AI_FALLBACK_KEY": "",
           "PEAKCUT_AI_FALLBACK_KEY_FILE": "", "PEAKCUT_AI_FALLBACK_BASE_URL": "",
           "PEAKCUT_UPLOAD_PART_MB": "5", "PEAKCUT_MAX_UPLOAD_MB": "64"}
    log = data / "server.log"
    failed = []
    with log.open("w", encoding="utf-8") as out:
        server = subprocess.Popen([sys.executable, "-m", "studio.server"], cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)
    try:
        for _ in range(90):
            try:
                urllib.request.urlopen(BASE + "/healthz", timeout=2)
                break
            except Exception:  # noqa: BLE001
                if server.poll() is not None:
                    print(log.read_text(encoding="utf-8", errors="replace")[-2000:])
                    return 1
                time.sleep(1)
        else:
            print("The test server did not start.")
            return 1
        runs = [("permissions", ["tests/test_permissions.py", BASE, str(ROOT), sys.executable]),
                ("abuse and limits", ["tests/test_abuse.py", BASE, str(ROOT)]),
                ("queue under load", ["tests/test_load_queue.py", str(ROOT)]),
                ("deleting and bulk actions", ["tests/test_delete.py"]),
                ("big uploads in pieces", ["tests/test_upload_parts.py", BASE]),
                ("AI services", ["tests/test_ai.py"]),
                ("no sign-in, and the AI key from the dashboard", ["tests/test_local.py"]),
                ("Hindi and Urdu captions in Roman letters", ["tests/test_roman.py"]),
                ("Portuguese captions, prompts and language selector", ["tests/test_ptbr.py"]),
                ("turning captions off and back on", ["tests/test_captions.py"]),
                ("connecting YouTube, Instagram and TikTok", ["tests/test_connect.py"]),
                ("the new-version notice", ["tests/test_updates.py"]),
                ("no limits by default", ["tests/test_limits.py"]),
                ("captions never overlap, stuck clips, sentence endings", ["tests/test_stuck.py"]),
                ("ad breaks and frozen screens", ["tests/test_static.py"]),
                ("answers that are not quite right", ["tests/test_answer.py"]),
                ("knowing which game a stream is", ["tests/test_games.py"]),
                ("finding the facecam in a gaming stream", ["tests/test_facecam.py"]),
                ("fresh installs: ffmpeg, the speech model, AI errors", ["tests/test_tools.py"]),
                ("the storage folder: full drive, open, move", ["tests/test_storage.py"]),
                ("security: visitor address, email addresses, Google sign-in, prompts", ["tests/test_hardening.py"]),
                ("long logs, the machine monitor, failed recordings, admin day totals", ["tests/test_robustness.py"]),
                ("Google Drive archiving and SSD cleanup", ["tests/test_gdrive.py"]),
                ("long videos, derived Shorts and magnetic thumbnails", ["tests/test_long_videos.py"]),
                ("long video series", ["tests/test_long_series.py"]),
                ("Home PT-BR, platform pause, 3-per-period cap and >5s music guard", ["tests/test_home_music_schedule.py"]),
                ("default covers, offline remote QR, light mode, mobile dock and perf", ["tests/test_cover_remote_perf.py"]),
                ("cache, re-cut and thumbnail variations (32 real tests)", ["tests/test_cache_recut_thumb.py"])]
        for name, args in runs:
            print(f"\n=== {name} ===", flush=True)
            r = subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
            lines = (r.stdout + r.stderr).splitlines()
            passed = sum(1 for x in lines if x.startswith("PASS"))
            for x in lines:
                if not x.startswith("PASS") and x.strip():
                    print("  " + x)
            print(f"  {passed} checks passed")
            if r.returncode != 0 or any(x.startswith("FAIL") for x in lines):
                failed.append(name)
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        _rm_tree(data)
    print("\nFAILED: " + ", ".join(failed) if failed else "\nAll tests passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
