from pydantic_settings import BaseSettings, SettingsConfigDict


def _resolve_minimax_key() -> str:
    """Resolve the MiniMax API key at access time (not import time).

    Resolution order:
      1. ``MINIMAX_API_KEY`` env var
      2. ``ANTHROPIC_AUTH_TOKEN`` env var (the same bearer Claude Code sets
         in its own session — convenient when developing from Claude Code)
    Reads the env at *call* time so terminals that don't inherit the
    Claude Code session still pick up keys the user exports.
    """
    import os
    return os.environ.get("MINIMAX_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # FPT AI Factory
    fpt_api_key: str
    fpt_base_url: str
    fpt_embedding_model: str = "vietnamese-embedding"
    fpt_llm_model: str = "Qwen3-32B"

    # MiniMax M3 (OpenAI-compatible) — used by RAGAS judge + answer generator
    # in scripts/evaluate_ragas.py. Set MINIMAX_API_KEY in your environment
    # (or in .env) before running; ANTHROPIC_AUTH_TOKEN also works.
    minimax_api_key: str = ""        # resolved via _resolve_minimax_key() below
    minimax_base_url: str = "https://api.minimax.io/v1"
    minimax_llm_model: str = "MiniMax-M3"
    disable_thinking: bool = True   # M3 is a reasoning model; default off for RAGAS

    # Retrieval
    top_k_dense: int = 100
    top_k_bm25: int = 100
    top_k_final: int = 10
    embedding_dim: int = 1024

    # Paths
    data_dir: str = "data"
    index_dir: str = "indexes"
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"


settings = Settings()

# MiniMax key is resolved lazily (see helper above) so terminal sessions that
# don't inherit the Claude Code env still work after the user exports the var.
settings.minimax_api_key = _resolve_minimax_key()
