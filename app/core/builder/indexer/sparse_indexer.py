from llama_index.core.node_parser import (
    MarkdownElementNodeParser,
    SimpleNodeParser,
)
from llama_index.core import VectorStoreIndex , Document
from app.core.builder.preprocessors.basic import BasicPreprocessor
from app.storage import DiskStore
from ...interface.base_indexer import BaseIndexer
from app.initialization import gemini_pro_model_llamaindex, gemini_embeddings_model_llamaindex
from app.logging_config import indexing_logger as logger    
import hashlib

index_source = "sparse"

class SparseIndexer(BaseIndexer):
    def __init__(self):
        self.markdown_parser = MarkdownElementNodeParser(
            num_workers=2,
            include_metadata=False
        )

    def index(self, index_name, documents):
        """Index documents using recursive text splitting and sentence window processing"""
        logger.info(f"Starting sparse indexing for index '{index_name}'")
        try:
            logger.debug("Processing documents into nodes")
            document_nodes , objects = self.create_chunks_from_documents(documents)
            
            logger.info(f"Building VectorStoreIndex from {len(document_nodes)} nodes")
            recursive_index = VectorStoreIndex(nodes=document_nodes + objects, embed_model =None, llm=gemini_pro_model_llamaindex, show_progress=True)
            
            logger.debug("Persisting index to storage")
            DiskStore.persist_index(recursive_index, index_source, index_name)
            return recursive_index
        
        except Exception as e:
            logger.error(f"Error during indexing for '{index_name}': {str(e)}", exc_info=True)
            raise

    def get_index_from_storage(self, index_name):
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

    def create_chunks_from_documents(self, documents):
        if not documents:
            raise ValueError("No documents provided for chunking")
        """Process markdown documents with subdoc batching"""
        try:
            nodes = []
            objects = []
            batch_size = 10
            
            # First split all documents into subdocs
            sub_docs = BasicPreprocessor.split_docs_by_separator(documents)
            logger.info(f"Total sub-documents created: {len(sub_docs)}")
            
            # Process subdocs in batches
            for i in range(0, len(sub_docs), batch_size):
                sub_docs_batch = sub_docs[i:i+batch_size]
                logger.info(f"Processing subdoc batch {i//batch_size+1}/{(len(sub_docs)-1)//batch_size+1}")
                
                # Process each subdoc batch
                batch_nodes = self.markdown_parser.get_nodes_from_documents(sub_docs_batch)
                base_nodes, batch_objects = self.markdown_parser.get_nodes_and_objects(batch_nodes)
                
                # Accumulate results
                nodes.extend(base_nodes)
                objects.extend(batch_objects)
                
                logger.debug(f"Processed {len(batch_nodes)} nodes in current batch")

            logger.info(f"Extracted total {len(nodes)} markdown elements from {len(objects)} objects")
            return nodes, objects
            
        except Exception as e:
            logger.error(f"Error processing markdown documents: {e}")
            raise

# Example usage
# analytical_indexer = AnalyticalIndexer()
# documents = analytical_indexer.create_documents_from_subdocs(sub_docs)
