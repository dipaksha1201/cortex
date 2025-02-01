from typing import Dict, Any
from langchain.retrievers import MultiVectorRetriever
from langchain.storage import LocalFileStore
from ..storage.pinecone import PineconeStore
from ..storage.disk_store import indexing_directory
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiVectorRetrieverBuilder:
    """Builder class for creating MultiVectorRetriever instances"""
    
    def __init__(self):
        self.pinecone_store = PineconeStore()
        self.storage_path = indexing_directory + "/vector_docstore"
        
    def build(self, index_name: str, id_key: str = "doc_id") -> MultiVectorRetriever:
        """
        Build a MultiVectorRetriever with the specified index name
        
        Args:
            index_name: Name of the Pinecone index to use
            id_key: Key to use for document IDs (default: "doc_id")
            
        Returns:
            MultiVectorRetriever instance
        """
        try:
            # Get vector store from Pinecone
            vector_store = self.pinecone_store.get_vector_store(index_name)
            
            # Initialize local file store
            store = LocalFileStore(root_path=self.storage_path + f"/{index_name}")
            
            # Create and return the retriever
            retriever = MultiVectorRetriever(
                vectorstore=vector_store,
                byte_store=store,
                id_key=id_key,
            )
            
            logger.info(f"MultiVectorRetriever created for index: {index_name}")
            return retriever
            
        except Exception as e:
            logger.error(f"Error creating MultiVectorRetriever: {e}")
            raise
