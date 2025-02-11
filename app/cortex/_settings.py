from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    pinecone_api_key: str = ""
    pinecone_index_name: str = "user-memory"
    pinecone_namespace: str = "memory"
    model: str = "models/gemini-pro"


SETTINGS = Settings()
