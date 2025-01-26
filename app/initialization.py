import logging
from config import gemini_pro_model, gemini_embeddings_model
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
import os
from llama_index.core import Settings
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMInitializer:
    def __init__(self):
        self.llm = self.initialize_llm()
        self.embedding_model = self.initialize_embedding_model()

    def initialize_llm(self , type):
        if type == "gemini_pro":
            # Initialize the LLM model using gemini_pro_model
            logger.info(f"Initializing LLM model: {gemini_pro_model}")
            llm = Gemini(model=gemini_pro_model, temperature=0.3, api_key=os.getenv("GEMINI_API_KEY_PROD"))
            Settings.llm = llm
            return llm
        elif type == "gemini_thinking":
            # Initialize the LLM model using gemini_pro_model
            logger.info(f"Initializing LLM model: {gemini_thinking_model}")
            gemini_llm = Gemini(model=gemini_thinking_model, temperature=0.3, api_key=os.getenv("GEMINI_API_KEY_PROD"))
            return gemini_llm

    def initialize_embedding_model(self):
        # Initialize the embedding model using gemini_embeddings_model
        logger.info(f"Initializing Embedding model: {gemini_embeddings_model}")
        gemini_embeddings = GeminiEmbedding(
            model_name=gemini_embeddings_model,
            api_key=os.getenv("GEMINI_API_KEY_PROD"),
        )
        Settings.embed_model = gemini_embeddings
        return gemini_embeddings

# Global instance of LLMInitializer
llm_initializer = LLMInitializer()
gemini_pro_model = llm_initializer.initialize_llm("gemini_pro")
# gemini_thinking_model = llm_initializer.initialize_llm("gemini_thinking")
gemini_embeddings_model = llm_initializer.initialize_embedding_model()
