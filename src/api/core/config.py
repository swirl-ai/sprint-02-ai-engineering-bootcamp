from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    OPENAI_API_KEY: str
    GROQ_API_KEY: str
    GOOGLE_API_KEY: str
    QDRANT_URL: str
    QDRANT_COLLECTION_NAME: str 
    EMBEDDING_MODEL: str
    EMBEDDING_MODEL_PROVIDER: str
    GENERATION_MODEL: str
    GENERATION_MODEL_PROVIDER: str
    LANGSMITH_TRACING: bool
    LANGSMITH_ENDPOINT: str
    LANGSMITH_API_KEY: str
    LANGSMITH_PROJECT: str
    RAG_PROMPT_TEMPLATE_PATH: str = "src/api/rag/prompts/rag_generation.yaml"

    model_config = SettingsConfigDict(env_file=".env")

class Settings(BaseSettings):

    DEFAULT_TIMEOUT: float = 30.0
    VERSION: str = "0.1.0"

config = Config()
settings = Settings()