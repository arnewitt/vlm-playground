from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_id: str = "vikhyatk/moondream2"
    revision: str = "2025-06-21"

    model_config = SettingsConfigDict(env_file=".env")
