"""Configuration for rag_vie.datagen — noise generation pipeline.

Reuses FPT credentials from the project's root .env for embedding validation.
Ollama settings are local-only (no API key needed).

Paths are resolved relative to the current working directory — run all
datagen commands from the project root (same convention as rag_vie.config).
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class GenDataSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Ollama (local LLM) ──────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:14b"
    ollama_temperature: float = 0.8
    ollama_num_ctx: int = 2048

    # ── FPT AI Factory (embedding for validation) ───────────────────────
    fpt_api_key: str
    fpt_base_url: str
    fpt_embedding_model: str = "Vietnamese_Embedding"

    # ── Validation ──────────────────────────────────────────────────────
    similarity_threshold: float = 0.85

    # ── Processing ──────────────────────────────────────────────────────
    max_retries: int = 3
    checkpoint_every: int = 100  # save progress every N items

    # ── Paths (relative to project root / cwd) ──────────────────────────
    output_dir: str = "data/generated"
    log_dir: str = "data/generated/logs"

    @property
    def raw_dir(self) -> Path:
        return Path(self.output_dir) / "raw"

    @property
    def validated_dir(self) -> Path:
        return Path(self.output_dir) / "validated"

    @property
    def logs_path(self) -> Path:
        return Path(self.log_dir)


settings = GenDataSettings()
