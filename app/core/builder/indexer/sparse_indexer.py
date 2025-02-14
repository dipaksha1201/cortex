from llama_index.core.node_parser import (
    MarkdownElementNodeParser,
    SimpleNodeParser,
)
from llama_index.core import VectorStoreIndex, Document
from app.core.builder.preprocessors.basic import BasicPreprocessor
from app.storage import DiskStore
from ...interface.base_indexer import BaseIndexer
from app.initialization import gemini_flash_model_llamaindex
from app.logging_config import indexing_logger as logger    
import hashlib
from typing import List, Tuple, Any

index_source = "sparse"

class SparseIndexer(BaseIndexer):
    def __init__(self):
        self.markdown_parser = MarkdownElementNodeParser(
            num_workers=2,
            include_metadata=False
        )

    def create_chunks_from_documents(self, documents):
        if not documents:
            raise ValueError("No documents provided for chunking")
        
        try:
            # First split all documents into subdocs
            sub_docs = BasicPreprocessor.split_docs_by_separator(documents)
            logger.info(f"Total sub-documents created: {len(sub_docs)}")
            
            all_base_nodes = []
            all_objects = []
            batch_size = 10
            
            # Process subdocs in batches of 10
            for i in range(0, len(sub_docs), batch_size):
                batch = sub_docs[i:i + batch_size]
                logger.debug(f"Processing batch {i//batch_size + 1}/{(len(sub_docs)-1)//batch_size + 1}")
                
                # Process current batch
                batch_nodes = self.markdown_parser.get_nodes_from_documents(batch)
                base_nodes, objects = self.markdown_parser.get_nodes_and_objects(batch_nodes)
                
                # Accumulate results
                all_base_nodes.extend(base_nodes)
                all_objects.extend(objects)
                logger.debug(f"Processed {len(base_nodes)} nodes in current batch")
            
            logger.info(f"Extracted total {len(all_base_nodes)} markdown elements from {len(all_objects)} objects")
            return all_base_nodes, all_objects
            
        except Exception as e:
            logger.error(f"Error processing markdown documents: {e}")
            raise

    def index(self, index_name: str, documents: List[Document]) -> bool:
        """Synchronous indexing method"""
        logger.info(f"Starting sparse indexing for index '{index_name}'")
        try:
            document_nodes, objects = self.create_chunks_from_documents(documents)
            
            logger.info(f"Building VectorStoreIndex from {len(document_nodes)} nodes")
            recursive_index = VectorStoreIndex(nodes=document_nodes + objects, embed_model=None, llm=gemini_flash_model_llamaindex, show_progress=True)
            
            logger.debug("Persisting index to storage")
            DiskStore.persist_index(recursive_index, index_source, index_name)
            return True
        
        except Exception as e:
            logger.error(f"Error during Sparse indexing for '{index_name}': {str(e)}", exc_info=True)
            raise

    def get_index_from_storage(self, index_name: str):
        """Load index from disk storage with error handling"""
        try:
            logger.info(f"Loading index '{index_name}' from storage")
            loaded_index = DiskStore.load_index(index_source, index_name)
            if not loaded_index:
                logger.warning(f"No index found for '{index_name}'")
                return None
            return loaded_index
        except Exception as e:
            logger.error(f"Failed to load index '{index_name}': {e}")
            raise

# Example usage
# analytical_indexer = AnalyticalIndexer()
# documents = analytical_indexer.create_documents_from_subdocs(sub_docs)
