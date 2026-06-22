import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Dataset Doctor Agent"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    UPLOAD_DIR: str = "uploads"
    OUTPUT_DIR: str = "outputs"
    GEMINI_API_KEY: str = ""
    ELEVENLABS_API_KEY: str = ""

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
