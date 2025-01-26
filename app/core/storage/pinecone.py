from pinecone import Pinecone, ServerlessSpec
import logging
from dotenv import load_dotenv
import os   
from ...initialization import gemini_embeddings_model

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PineconeStore:
    def __init__(self, api_key, environment):
        self.initialized_pinecone = Pinecone(api_key=os.getenv("PINE_CONE_API_KEY"))
        logger.info("Pinecone initialized")

    def get_index(self, index_name):
        if index_name not in self.initialized_pinecone.list_indexes():
            self.initialized_pinecone.create_index(
                name=index_name,
                dimension=768, # Replace with your model dimensions
                metric="cosine", # Replace with your model metric
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                ) 
            )
            logger.info(f"Index {index_name} created")
            return self.initialized_pinecone.Index(index_name)
        else:
            logger.info(f"Index {index_name} already exists")
            return self.initialized_pinecone.Index(index_name)

    def get_vector_store(self, index_name):
        return PineconeVectorStore(index=self.get_index(index_name), embedding=gemini_embeddings_model)

# Example usage
# pinecone_store = PineconeStore(api_key='your_api_key', environment='us-west1')
# pinecone_store.create_index('my_index')
# vector_store = pinecone_store.get_vector_store('my_index')