
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables from .env
load_dotenv()


class Settings(BaseSettings):
    # Google Gemini
    gemini_api_key: str = ""
    gemini_chat_model: str = "gemini-2.5-flash"
    gemini_embed_model: str = "gemini-embedding-001"
    gemini_vision_model: str = "gemini-2.5-flash"

    # Pinecone Vector Database
    pinecone_api_key: str = ""
    pinecone_index: str = "medical-agent-demo"

    # Tavily Web Search
    tavily_api_key: str = ""

    # API Security
    api_token: str = ""

    # CORS
    allowed_origins: str = (
        "http://localhost:8000,"
        "http://127.0.0.1:8000"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
