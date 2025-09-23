"""Global configuration using Pydantic settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application wide settings loaded from environment variables."""

    base_dir: Path = Field(default_factory=lambda: Path("recordings"))
    database_path: Path = Field(default_factory=lambda: Path("adsum.db"))
    sample_rate: int = 16_000
    channels: int = 1
    chunk_seconds: float = 1.0
    audio_backend: str = "ffmpeg"
    ffmpeg_binary: str = "ffmpeg"
    ffmpeg_download_url: Optional[str] = None
    openai_transcription_model: str = "gpt-4o-mini-transcribe"
    openai_notes_model: str = "gpt-4o-mini"
    openai_api_key: Optional[str] = None
    session_prefix: str = "session"
    default_mic_device: Optional[str] = None
    default_system_device: Optional[str] = None

    model_config = SettingsConfigDict(
        env_prefix="ADSUM_",
        env_file=".env",
        case_sensitive=False,
    )

    @property
    def base_dir_raw(self) -> Path:
        path = self.base_dir / "raw"
        path.mkdir(parents=True, exist_ok=True)
        return path


_settings: Optional[Settings] = None


_MODEL_CONFIG: Dict[str, Any] = dict(Settings.model_config or {})
_ENV_PREFIX: str = (_MODEL_CONFIG.get("env_prefix") or "").upper()
_CASE_SENSITIVE: bool = bool(_MODEL_CONFIG.get("case_sensitive", True))
_ENV_FILE = _MODEL_CONFIG.get("env_file") or ".env"
_ENV_PATH = Path(_ENV_FILE)


@dataclass
class EnvironmentSetting:
    """Metadata about a configuration option backed by an environment variable."""

    field: str
    env_name: str
    value: Any
    default: Any
    annotation: Any


class EnvironmentSettingError(RuntimeError):
    """Raised when environment-backed configuration updates fail."""


def _env_key(field: str) -> str:
    key = f"{_ENV_PREFIX}{field}" if _ENV_PREFIX else field
    return key if _CASE_SENSITIVE else key.upper()


def _field_default(field_info) -> Any:
    if field_info.default is not None:
        return field_info.default
    if field_info.default_factory is not None:  # type: ignore[truthy-function]
        return field_info.default_factory()
    return None


def _load_env_file() -> Iterable[str]:
    if not _ENV_PATH.exists():
        return []
    return _ENV_PATH.read_text().splitlines()


def _persist_env_value(env_name: str, value: Optional[str]) -> None:
    lines = list(_load_env_file())
    updated = False
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            new_lines.append(line)
            continue
        key, current = line.split("=", 1)
        if key.strip() == env_name:
            updated = True
            if value is None:
                continue
            new_lines.append(f"{env_name}={value}")
        else:
            new_lines.append(line)
    if not updated and value is not None:
        new_lines.append(f"{env_name}={value}")

    if new_lines:
        _ENV_PATH.write_text("\n".join(new_lines) + "\n")
    else:
        if _ENV_PATH.exists():
            _ENV_PATH.unlink()


def list_environment_settings(settings: Optional[Settings] = None) -> Iterable[EnvironmentSetting]:
    """Return metadata for all environment-backed settings."""

    settings = settings or get_settings()
    for name, field in Settings.model_fields.items():
        env_name = _env_key(name)
        value = getattr(settings, name)
        default = _field_default(field)
        yield EnvironmentSetting(
            field=name,
            env_name=env_name,
            value=value,
            default=default,
            annotation=field.annotation,
        )


def _apply_setting_update(field: str, raw_value: Optional[str]) -> Settings:
    env_name = _env_key(field)
    previous = os.environ.get(env_name)

    if raw_value is None:
        os.environ.pop(env_name, None)
    else:
        os.environ[env_name] = raw_value

    try:
        new_settings = Settings()
    except ValidationError as exc:  # pragma: no cover - requires invalid input
        if previous is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = previous
        raise EnvironmentSettingError(str(exc)) from exc

    global _settings
    _settings = new_settings
    _persist_env_value(env_name, raw_value)
    return new_settings


def update_environment_setting(field: str, raw_value: str) -> Settings:
    """Update an environment setting and reload configuration."""

    return _apply_setting_update(field, raw_value)


def clear_environment_setting(field: str) -> Settings:
    """Remove an environment override for the given field and reload configuration."""

    return _apply_setting_update(field, None)


def get_settings() -> Settings:
    """Return a singleton instance of the application settings."""

    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


__all__ = [
    "Settings",
    "EnvironmentSetting",
    "EnvironmentSettingError",
    "clear_environment_setting",
    "get_settings",
    "list_environment_settings",
    "update_environment_setting",
]

