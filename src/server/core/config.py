from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_id: str = "vikhyatk/moondream2"
    revision: str = "2025-06-21"
    host: str = "0.0.0.0"
    port: int = 9999
    app_title: str = "VLM API"

    model_config = SettingsConfigDict(env_file=".env")
