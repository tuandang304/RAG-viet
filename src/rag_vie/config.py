from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # FPT AI Factory
    fpt_api_key: str
    fpt_base_url: str
    fpt_embedding_model: str = "vietnamese-embedding"
    fpt_sparse_model: str = "vietnamese-embedding"   # thường cùng model với dense
    fpt_llm_model: str = "Qwen3-32B"

    # Retrieval
    top_k_dense: int = 100
    top_k_bm25: int = 100
    top_k_sparse: int = 100
    top_k_final: int = 10
    embedding_dim: int = 1024

    # Paths
    data_dir: str = "data"
    index_dir: str = "indexes"
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"


settings = Settings()
