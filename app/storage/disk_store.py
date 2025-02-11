import logging
from llama_index.core import load_index_from_storage , ServiceContext
from llama_index.core import StorageContext
from app.initialization import gemini_embeddings_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

indexing_directory = "/Users/dipak/CortexProjects/storage"

class DiskStore:
    @staticmethod
    def persist_index(index,indexing_source, index_name):
        try:
            index.storage_context.persist(f"{indexing_directory}/{indexing_source}/{index_name}")
            logger.info(f"Index created and saved as {index_name}")
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            return False    

    @staticmethod
    def load_index(indexing_source,index_name):
        try:
            loaded_storage = StorageContext.from_defaults(
                persist_dir=f"{indexing_directory}/{indexing_source}/{index_name}"
            )
            # service_context = ServiceContext.from_defaults(embed_model=gemini_embeddings_model)
            loaded_index = load_index_from_storage(loaded_storage)
            logger.info(f"Index loaded from storage: {index_name}")
            return loaded_index
        except Exception as e:
            logger.error(f"Error loading index from storage: {e}")
            return False


# Example usage
# storage = IndexStorage()
# storage.persist_index(index, "my_index", "/path/to/indexing_directory")
# storage.log_persist_action(index, "my_index", "/path/to/indexing_directory")
# loaded_index = IndexStorage.load_index("my_index", "my_storage")
