from pydantic_settings import BaseSettings
from functools import lru_cache

# Global Model Configurations
gemini_pro_model = "models/gemini-pro"
gemini_embeddings_model = "models/embedding-001"
gemini_thinking_model = "gemini-2.0-flash-thinking-exp"

class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Cortex"
    
    # Model Configurations
    # gemini_pro_model: str = gemini_pro_model
    # gemini_embeddings_model: str = gemini_embeddings_model
    
    # Security
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Database
    DATABASE_URL: str = "sqlite:///./cortex.db"  # Default SQLite, change as needed
    
    class Config:
        case_sensitive = True
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
