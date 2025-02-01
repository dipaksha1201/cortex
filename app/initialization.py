import logging
from .config import gemini_pro_model, gemini_embeddings_model, gemini_pro_model_langchain
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai
import os
from llama_index.core import Settings
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMInitializer:

    def initialize_langchain_llm(self , type):
        if type == "gemini_pro":
            # Initialize the LLM model using gemini_pro_model
            logger.info(f"Initializing LLM model: {gemini_pro_model}")
            llm = ChatGoogleGenerativeAI(model=gemini_pro_model_langchain, google_api_key = os.getenv("GEMINI_API_KEY_PROD"))
            return llm
        
    def initialize_llamaindex_llm(self , type):
        if type == "gemini_pro":
            # Initialize the LLM model using gemini_pro_model
            logger.info(f"Initializing LLM model: {gemini_pro_model}")
            llm = Gemini(model=gemini_pro_model, temperature=0.3, api_key=os.getenv("GEMINI_API_KEY_PROD"))
            Settings.llm = llm
            return llm
        
    def initialize_langchain_embedding_model(self):
        # Initialize the LLM model using gemini_pro_model
        logger.info(f"Initializing langchain embedding model: {gemini_pro_model}")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key = os.getenv("GEMINI_API_KEY_PROD"))
        return embeddings

    def initialize_llamaindex_embedding_model(self):    
        # Initialize the embedding model using gemini_embeddings_model
        logger.info(f"Initializing Embedding model: {gemini_embeddings_model}")
        gemini_embeddings = GeminiEmbedding(
            model_name=gemini_embeddings_model,
            api_key=os.getenv("GEMINI_API_KEY_PROD"),
        )
        Settings.embed_model = gemini_embeddings
        return gemini_embeddings

    def initialize_google_client(self):
        # Initialize the LLM model using gemini_pro_model
        logger.info(f"Initializing LLM model: {gemini_pro_model}")
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY_PROD"), http_options={'api_version':'v1alpha'})
        return client

# Global instance of LLMInitializer
llm_initializer = LLMInitializer()
gemini_pro_model = llm_initializer.initialize_llamaindex_llm("gemini_pro")
google_client = llm_initializer.initialize_google_client()
gemini_embeddings_model = llm_initializer.initialize_llamaindex_embedding_model()
gemini_pro_model_langchain = llm_initializer.initialize_langchain_llm("gemini_pro")
gemini_langchain_embeddings = llm_initializer.initialize_langchain_embedding_model()