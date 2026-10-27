from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables and an optional `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+psycopg://crisk:crisk@localhost:5434/crisk"
    test_database_url: str | None = None
    kaggle_api_token: SecretStr | None = None
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()
