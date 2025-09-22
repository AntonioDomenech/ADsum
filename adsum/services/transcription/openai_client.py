"""OpenAI powered transcription service."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ...config import get_settings
from ...data.models import RecordingSession, TranscriptResult, TranscriptSegment
from ...logging import get_logger
from .base import TranscriptionService

LOGGER = get_logger(__name__)


class OpenAITranscriptionService(TranscriptionService):
    def __init__(self, model: Optional[str] = None) -> None:
        settings = get_settings()
        self.model = model or settings.openai_transcription_model
        try:
            from openai import OpenAI, OpenAIError  # type: ignore
        except ImportError as exc:  # pragma: no cover - runtime dependency guard
            raise RuntimeError("openai package is required for OpenAITranscriptionService") from exc
        client_kwargs = {}
        if settings.openai_api_key:
            client_kwargs["api_key"] = settings.openai_api_key

        try:
            self.client = OpenAI(**client_kwargs)
        except OpenAIError as exc:
            message = str(exc)
            if "api_key" in message.lower():
                raise RuntimeError(
                    "OpenAI API key not configured. Set the OPENAI_API_KEY environment variable "
                    "or configure ADSUM_OPENAI_API_KEY from the Environment menu."
                ) from exc
            raise RuntimeError(f"Failed to initialise OpenAI transcription client: {message}") from exc

    def transcribe(self, session: RecordingSession, audio_path: Path) -> TranscriptResult:
        LOGGER.info("Requesting OpenAI transcription for %s", audio_path)
        with open(audio_path, "rb") as audio_file:
            response = self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                response_format="verbose_json",
            )
        segments = [
            TranscriptSegment(start=segment["start"], end=segment["end"], text=segment["text"])
            for segment in response.get("segments", [])
        ]
        text = response.get("text", "")
        return TranscriptResult(
            session_id=session.id,
            channel=audio_path.stem,
            text=text,
            segments=segments,
            raw_response=response,
        )

    def transcribe_stream(
        self,
        session: RecordingSession,
        audio_path: Path,
        on_update: Optional[Callable[[TranscriptResult], None]] = None,
    ) -> TranscriptResult:
        if on_update is None:
            return self.transcribe(session, audio_path)

        streaming = getattr(self.client.audio.transcriptions, "with_streaming_response", None)
        if streaming is None:
            LOGGER.info("OpenAI client does not support streaming responses; using batch transcription")
            return self.transcribe(session, audio_path)

        channel = audio_path.stem
        text_parts: List[str] = []
        incremental_segments: List[str] = []
        last_payload: Optional[TranscriptResult] = None

        def emit_partial() -> None:
            nonlocal last_payload
            text = "".join(text_parts).strip()
            segments = [
                TranscriptSegment(text=segment.strip())
                for segment in incremental_segments
                if segment.strip()
            ]
            if not text and not segments:
                return
            last_payload = TranscriptResult(
                session_id=session.id,
                channel=channel,
                text=text,
                segments=segments,
            )
            try:
                on_update(last_payload)
            except Exception:  # pragma: no cover - callbacks should not break pipeline
                LOGGER.exception("Transcript update callback raised an exception")

        LOGGER.info("Requesting OpenAI streaming transcription for %s", audio_path)
        parsed: object | None = None
        try:
            with open(audio_path, "rb") as audio_file:
                with streaming.create(
                    model=self.model,
                    file=audio_file,
                    response_format="verbose_json",
                    stream=True,
                ) as response:
                    for line in response.iter_lines():
                        if not line or line.startswith(":"):
                            continue
                        if line.strip().upper() == "DATA: [DONE]":
                            break
                        if not line.startswith("data:"):
                            continue
                        payload = line[len("data:") :].strip()
                        if not payload or payload == "[DONE]":
                            continue
                        try:
                            data: Dict[str, object] = json.loads(payload)
                        except json.JSONDecodeError:
                            LOGGER.debug("Failed to decode streaming payload: %s", payload)
                            continue
                        event_type = str(data.get("type", ""))
                        if event_type == "transcript.text.delta":
                            delta = str(data.get("delta", ""))
                            if delta:
                                text_parts.append(delta)
                                incremental_segments.append(delta)
                                emit_partial()
                        elif event_type == "transcript.text.done":
                            text = str(data.get("text", ""))
                            if text:
                                text_parts.append(text)
                                incremental_segments.append(text)
                                emit_partial()
                        elif event_type == "response.completed":
                            break

                    parsed = response.parse()
        except Exception as exc:  # pragma: no cover - requires network failures
            LOGGER.exception("Streaming transcription failed; falling back to batch mode: %s", exc)
            return self.transcribe(session, audio_path)

        segments: List[TranscriptSegment] = []
        text = "".join(text_parts).strip()
        raw_response = None
        try:
            from openai.types.audio.transcription_verbose import TranscriptionVerbose

            if isinstance(parsed, TranscriptionVerbose):
                raw_response = parsed.model_dump()
                text = parsed.text or text
                for segment in parsed.segments or []:
                    segments.append(
                        TranscriptSegment(
                            start=getattr(segment, "start", None),
                            end=getattr(segment, "end", None),
                            text=(getattr(segment, "text", "") or "").strip(),
                        )
                    )
            elif hasattr(parsed, "model_dump"):
                raw_response = parsed.model_dump()
                text = raw_response.get("text", text)
                for segment in raw_response.get("segments", []) or []:
                    segments.append(
                        TranscriptSegment(
                            start=segment.get("start"),
                            end=segment.get("end"),
                            text=(segment.get("text") or "").strip(),
                        )
                    )
        except Exception:  # pragma: no cover - best effort parsing
            LOGGER.debug("Failed to parse streaming response payload", exc_info=True)

        final_result = TranscriptResult(
            session_id=session.id,
            channel=channel,
            text=text,
            segments=segments,
            raw_response=raw_response,
        )

        if last_payload is None or last_payload.model_dump() != final_result.model_dump():
            try:
                on_update(final_result)
            except Exception:  # pragma: no cover - callbacks should not break pipeline
                LOGGER.exception("Transcript update callback raised an exception")

        return final_result


__all__ = ["OpenAITranscriptionService"]

