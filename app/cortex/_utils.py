from __future__ import annotations

from functools import lru_cache

import langsmith
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from langchain_google_genai import ChatGoogleGenerativeAI
from lang_memgpt import _schemas as schemas
from lang_memgpt import _settings as settings
from app.storage.pinecone import PineconeStore
import os

_DEFAULT_DELAY = 60  # seconds

class utils:
    
    @staticmethod
    def get_index():
        store = PineconeStore()
        index = store.get_index(settings.SETTINGS.pinecone_index_name)
        return index

    @staticmethod
    def ensure_configurable(config: RunnableConfig) -> schemas.GraphConfig:
        """Merge the user-provided config with default values."""
        configurable = config.get("configurable", {})
        return {
            **configurable,
            **schemas.GraphConfig(
                delay=configurable.get("delay", _DEFAULT_DELAY),
                model=configurable.get("model", settings.SETTINGS.model),
                thread_id=configurable["thread_id"],
                user_id=configurable["user_id"],
            ),
        }


    @lru_cache
    @staticmethod
    def get_embeddings():
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key = os.getenv("GEMINI_API_KEY"))

    @staticmethod
    def get_llm(model):
        return ChatGoogleGenerativeAI(model=model, google_api_key = os.getenv("GEMINI_API_KEY_PROD"))

