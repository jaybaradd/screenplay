from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _content_mode() -> str:
    value = os.getenv("LANGFUSE_CONTENT_MODE", "preview").lower()
    return value if value in {"preview", "metadata", "full"} else "preview"


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("DATA_DIR", "./data")).resolve()
    ai_mode: str = os.getenv("AI_MODE", "mock")
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY") or None
    text_model: str = os.getenv("GEMINI_TEXT_MODEL", "gemini-3.8-flash")
    image_model: str = os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image")
    langfuse_content_mode: str = _content_mode()
    langfuse_preview_chars: int = _int("LANGFUSE_PREVIEW_CHARS", 8000)
    langfuse_environment: str = os.getenv("LANGFUSE_ENVIRONMENT", "development")
    langfuse_release: str = os.getenv("LANGFUSE_RELEASE", "0.3.0")
    gemini_pricing_tier: str = os.getenv("GEMINI_PRICING_TIER", "paid_standard")
    api_base_url: str = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

    @property
    def app_db(self) -> Path:
        return self.data_dir / "app.sqlite"

    @property
    def checkpoint_db(self) -> Path:
        return self.data_dir / "checkpoints.sqlite"

    @property
    def asset_dir(self) -> Path:
        return self.data_dir / "assets"

    @property
    def export_dir(self) -> Path:
        return self.data_dir / "exports"

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.asset_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        font_cache = self.data_dir / ".cache"
        font_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("XDG_CACHE_HOME", str(font_cache))

    @property
    def langfuse_capture_content(self) -> bool:
        """Compatibility alias for older callers; full capture is no longer the default."""
        return self.langfuse_content_mode == "full"


settings = Settings()
