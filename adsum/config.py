"""Global configuration using Pydantic settings."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application wide settings loaded from environment variables."""

    base_dir: Path = Field(default_factory=lambda: Path("recordings"))
    database_path: Path = Field(default_factory=lambda: Path("adsum.db"))
    sample_rate: int = 16_000
    channels: int = 1
    chunk_seconds: float = 1.0
    openai_transcription_model: str = "gpt-4o-mini-transcribe"
    openai_notes_model: str = "gpt-4o-mini"
    session_prefix: str = "session"

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


def get_settings() -> Settings:
    """Return a singleton instance of the application settings."""

    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


__all__ = ["Settings", "get_settings"]

