from ....initialization import gemini_pro_model
from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor,
    SimpleLLMPathExtractor,
)
from llama_index.core import PropertyGraphIndex
from ..preprocessors import BasicPreprocessor
import logging
from ...storage import DiskStore
from ...interface.base_indexer import BaseIndexer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

index_source = "knowledge_graph"
class KnowledgeGraphIndexer(BaseIndexer):
    def __init__(self):
        pass

    def index(self, index_name, documents):
        try:
            index = self.get_index_from_storage(index_name)
            sub_docs = BasicPreprocessor.split_docs_by_separator(documents)
            
            if index:
                index.insert(sub_docs)
            else:
                # Implement knowledge graph indexing logic here
                index = self.create_property_graph_index(sub_docs, gemini_pro_model)

            DiskStore.persist_index(index,index_source, index_name)
            return True
        except Exception as e:
            logger.error(f"Error in indexing: {e}")
            return False

    def get_index_from_storage(self, index_name):    
        try:
            loaded_index = DiskStore.load_index(index_source,index_name)
            return loaded_index
        except Exception as e:
            logger.error(f"Error loading index from storage: {e}")
            return False
    
    def create_property_graph_index(self, sub_docs, llm):
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
            )
            logger.info("Property graph index created")
            return index
        except Exception as e:
            logger.error(f"Error creating property graph index: {e}")
            raise

# Example usage
# kg_indexer = KnowledgeGraphIndexer()
# sub_docs = kg_indexer.split_docs_by_separator(documents)
# index = kg_indexer.create_property_graph_index(sub_docs, gemini_llm)
# loaded_index = kg_indexer.get_index_from_storage(True)
