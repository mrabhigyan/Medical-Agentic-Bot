from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    gemini_api_key: str = ""
    gemini_chat_model: str = "gemini-3.8-flash"
    gemini_embed_model: str = "gemini-embedding-001"

    pinecone_api_key: str = ""
    pinecone_index: str = "medical-agent-demo"

    tavily_api_key: str = ""

    api_token: str = ""
    allowed_origins: str = (
        "http://localhost:8000,http://127.0.0.1:8000"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()