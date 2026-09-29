from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("DATA_DIR", "./data")).resolve()
    ai_mode: str = os.getenv("AI_MODE", "mock")
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY") or None
    text_model: str = os.getenv("GEMINI_TEXT_MODEL", "gemini-3.8-flash")
    image_model: str = os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image")
    langfuse_capture_content: bool = _bool("LANGFUSE_CAPTURE_CONTENT", False)
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


settings = Settings()
