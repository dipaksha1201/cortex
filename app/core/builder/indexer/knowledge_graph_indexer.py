from ....initialization import gemini_pro_model , gemini_embeddings_model
from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor,
    SimpleLLMPathExtractor,
)
from llama_index.core import PropertyGraphIndex
from llama_index.core.node_parser import SimpleNodeParser
from ..preprocessors import BasicPreprocessor
import logging
from app.storage import DiskStore
from ...interface.base_indexer import BaseIndexer
import nest_asyncio
nest_asyncio.apply()


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

index_source = "knowledge_graph"
class KnowledgeGraphIndexer(BaseIndexer):
        
    def __init__(self):
        pass

    def index(self, index_name, documents):
        logger.info(f"Starting KG indexing for index_name: {index_name}")
        try:
            logger.debug("Retrieving index from storage")
            index = self.get_index_from_storage(index_name)
            logger.debug("Splitting documents by separator")
            sub_docs = BasicPreprocessor.split_docs_by_separator(documents)
            logger.info(f"Number of sub-documents created: {len(sub_docs)}")
            
            if index:
                logger.info(f"Index found for '{index_name}'. Inserting sub-documents.")
                self.update_property_graph_index(index.property_graph_store, sub_docs, gemini_pro_model)
                
            else:
                logger.info(f"No existing index found for '{index_name}'. Creating new property graph index.")
                # index = await self.create_property_graph_index(sub_docs, gemini_pro_model)
                index = self.create_property_graph_index(sub_docs, gemini_pro_model)
                logger.info("New property graph index created successfully.")

            logger.info("Persisting index to disk storage")
            DiskStore.persist_index(index, index_source, index_name)
            logger.info(f"Index '{index_name}' persisted successfully.")
            return True
        except Exception as e:
            logger.error(f"Error in indexing for '{index_name}': {e}", exc_info=True)
            return False

    def insert_into_index(self, index, sub_docs):
        node_parser = SimpleNodeParser()
        nodes = node_parser.get_nodes_from_documents(sub_docs)
        logger.debug(f"Nodes: {nodes}")
        index.insert_nodes(nodes)
    
    def update_property_graph_index(self, store, sub_docs, llm):
        logger.info("Updating property graph index from documents")
        try:
            index = PropertyGraphIndex.from_documents(
                sub_docs,
                kg_extractors=[
                    ImplicitPathExtractor(),
                    SimpleLLMPathExtractor(
                        llm=llm,
                        num_workers=4,
                        max_paths_per_chunk=10,
                    ),
                ],
                show_progress=True,
                property_graph_store=store,
                embed_model=gemini_embeddings_model,
            )
            logger.info("Property graph index created successfully")
            return index
        except Exception as e:
            logger.error(f"Error creating property graph index: {e}", exc_info=True)
            raise
    
    def create_property_graph_index(self, sub_docs, llm):
        logger.debug("Creating property graph index from documents")
        try:
            index = PropertyGraphIndex.from_documents(
                sub_docs,
                kg_extractors=[
                    ImplicitPathExtractor(),
                    SimpleLLMPathExtractor(
                        llm=llm,
                        num_workers=4,
                        max_paths_per_chunk=10,
                    ),
                ],
                show_progress=True,
                embed_model=gemini_embeddings_model,
            )
            logger.info("Property graph index created successfully")
            return index
        except Exception as e:
            logger.error(f"Error creating property graph index: {e}", exc_info=True)
            raise

    def get_index_from_storage(self, index_name):    
        logger.info(f"Loading index '{index_name}' from storage")
        try:
            loaded_index = DiskStore.load_index(index_source, index_name)
            if loaded_index:
                logger.info(f"Index '{index_name}' loaded successfully from storage.")
            else:
                logger.info(f"Index '{index_name}' not found in storage.")
            return loaded_index
        except Exception as e:
            logger.info(f"Error loading index '{index_name}' from storage: {e}", exc_info=True)
            return False

