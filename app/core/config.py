from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "retail-router-executor"
    environment: str = "dev"
    api_v1_prefix: str = "/api/v1"

    database_url: str = f"sqlite:///{BASE_DIR / 'artifacts' / 'retail_router.db'}"
    raw_data_path: str = str(BASE_DIR / "online_retail.xlsx")
    artifacts_dir: str = str(BASE_DIR / "artifacts")

    default_forecast_horizon: int = 4
    default_product_clusters: int = 4
    default_customer_clusters: int = 4
    default_target_metric: str = "revenue"
    min_history_weeks: int = 8
    lag_weeks: int = 8
    max_entities_for_dtw: int = 250
    random_state: int = 42

    llm_provider: str = "heuristic"
    llm_model: str | None = None
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
Path(settings.artifacts_dir).mkdir(parents=True, exist_ok=True)

