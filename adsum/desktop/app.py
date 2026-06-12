"""Local desktop host for ADsum v2."""

from __future__ import annotations

import json
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from uuid import uuid4

from ..config import EnvironmentSettingError, get_settings, update_environment_setting
from ..data.models import RecordingSession
from ..logging import get_logger
from ..services.factory import resolve_transcription_backend
from .audio import NativeRecordingManager, list_desktop_audio_devices

LOGGER = get_logger(__name__)

def _static_dir() -> Path:
    bundled_root = getattr(sys, "_MEIPASS", None)
    if bundled_root:
        bundled_path = Path(bundled_root) / "adsum" / "desktop" / "static"
        if bundled_path.exists():
            return bundled_path
    return Path(__file__).parent / "static"


STATIC_DIR = _static_dir()


@dataclass
class DesktopServer:
    url: str
    httpd: ThreadingHTTPServer
    thread: threading.Thread

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)


class DesktopApi:
    def __init__(self) -> None:
        self.recordings = NativeRecordingManager()
        self._last_transcript: Optional[Dict[str, Any]] = None

    def devices(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"devices": list_desktop_audio_devices()}

    def status(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        settings = get_settings()
        return {
            "recording": self.recordings.status(),
            "openai_key_configured": bool(settings.openai_api_key),
            "last_transcript": self._last_transcript,
        }

    def save_openai_key(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        value = str(payload.get("key") or "").strip()
        if not value:
            raise ValueError("Paste a key before saving.")
        try:
            update_environment_setting("openai_api_key", value)
        except EnvironmentSettingError as exc:
            raise ValueError(str(exc)) from exc
        return {"ok": True}

    def start(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        name = str(payload.get("name") or "").strip()
        microphone_id = _optional_string(payload.get("microphone_id"))
        speaker_id = _optional_string(payload.get("speaker_id"))
        return {"recording": self.recordings.start(name=name, microphone_id=microphone_id, speaker_id=speaker_id)}

    def stop(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        result = self.recordings.stop()
        return {"result": result.to_dict(), "recording": self.recordings.status()}

    def test(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        microphone_id = _optional_string(payload.get("microphone_id"))
        speaker_id = _optional_string(payload.get("speaker_id"))
        duration = float(payload.get("duration_seconds") or 6.0)
        result = self.recordings.run_device_test(
            microphone_id=microphone_id,
            speaker_id=speaker_id,
            duration_seconds=duration,
        )
        return {"result": result.to_dict(), "recording": self.recordings.status()}

    def transcribe(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        result = self.recordings.last_result
        if result is None or result.paths.mixed_path is None:
            raise ValueError("Record or test audio before transcribing.")
        audio_path = result.paths.mixed_path
        if not audio_path.exists():
            raise ValueError(f"Audio file was not found: {audio_path}")

        service = resolve_transcription_backend("openai")
        if service is None:
            raise ValueError("OpenAI transcription is not available.")

        session = RecordingSession(
            id=str(uuid4()),
            name=result.name,
            created_at=result.started_at,
            duration=result.duration_seconds,
            sample_rate=16000,
            channels=1,
            audio_paths={"mixed": audio_path},
            mix_path=audio_path,
        )
        transcript = service.transcribe(session, audio_path)
        self._last_transcript = {
            "session_id": transcript.session_id,
            "channel": transcript.channel,
            "text": transcript.text,
            "segments": [segment.model_dump() for segment in transcript.segments],
            "created_at": time.time(),
        }
        return {"transcript": self._last_transcript}


def run_desktop_app(*, port: int = 0, browser: bool = False) -> None:
    api = DesktopApi()
    server = start_desktop_server(api, port=port)
    LOGGER.info("ADsum desktop server listening on %s", server.url)

    if not browser:
        try:
            import webview  # type: ignore
        except Exception as exc:
            LOGGER.warning("pywebview is unavailable; opening browser fallback: %s", exc)
        else:
            try:
                webview.create_window("ADsum", server.url, width=1180, height=780, min_size=(980, 640))
                webview.start()
                server.stop()
                return
            except Exception as exc:
                LOGGER.warning("Unable to open desktop webview; falling back to browser: %s", exc)

    webbrowser.open(server.url)
    print(f"ADsum desktop is running at {server.url}")
    print("Press Ctrl+C to stop the local server.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()


def start_desktop_server(api: DesktopApi, *, port: int = 0) -> DesktopServer:
    routes: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
        "/api/devices": api.devices,
        "/api/status": api.status,
        "/api/settings/openai-key": api.save_openai_key,
        "/api/start": api.start,
        "/api/stop": api.stop,
        "/api/test": api.test,
        "/api/transcribe": api.transcribe,
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/api/"):
                self._send_json(_dispatch_get(self.path))
                return
            self._serve_static()

        def do_POST(self) -> None:  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length") or "0")
                payload = {}
                if length:
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                handler = routes.get(self.path)
                if handler is None:
                    self._send_json({"error": "Unknown endpoint."}, status=HTTPStatus.NOT_FOUND)
                    return
                self._send_json(handler(payload))
            except Exception as exc:
                LOGGER.exception("Desktop API error at %s: %s", self.path, exc)
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

        def log_message(self, format: str, *args: Any) -> None:
            LOGGER.debug("desktop server: " + format, *args)

        def _serve_static(self) -> None:
            requested = self.path.split("?", 1)[0].lstrip("/")
            if not requested:
                requested = "index.html"
            target = (STATIC_DIR / requested).resolve()
            static_root = STATIC_DIR.resolve()
            if static_root not in target.parents and target != static_root:
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            if not target.exists() or not target.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content_type = _content_type(target)
            body = target.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: Dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def _dispatch_get(path: str) -> Dict[str, Any]:
        handler = routes.get(path.split("?", 1)[0])
        if handler is None:
            return {"error": "Unknown endpoint."}
        return handler({})

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    actual_port = int(httpd.server_address[1])
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return DesktopServer(url=f"http://127.0.0.1:{actual_port}/", httpd=httpd, thread=thread)


def _content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".html":
        return "text/html; charset=utf-8"
    if suffix == ".css":
        return "text/css; charset=utf-8"
    if suffix == ".js":
        return "application/javascript; charset=utf-8"
    if suffix == ".svg":
        return "image/svg+xml"
    return "application/octet-stream"


def _optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = ["DesktopApi", "run_desktop_app", "start_desktop_server"]
