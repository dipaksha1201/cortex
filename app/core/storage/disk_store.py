import logging
from llama_index.core import load_index_from_storage
from llama_index.core import StorageContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

indexing_directory = "/Users/dipak/CortexProjects/storage"

class DiskStore:
    @staticmethod
    def persist_index(index,indexing_source, file_name):
        try:
            index.storage_context.persist(f"{indexing_directory}/{indexing_source}/{file_name}")
            logger.info(f"Index created and saved as {file_name}")
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            raise

    @staticmethod
    def load_index(indexing_source,file_name):
        try:
            loaded_storage = StorageContext.from_defaults(
                persist_dir=f"{indexing_directory}/{indexing_source}/{file_name}"
            )
            loaded_index = load_index_from_storage(loaded_storage)
            logger.info(f"Index loaded from storage: {file_name}")
            return loaded_index
        except Exception as e:
            logger.error(f"Error loading index from storage: {e}")
            raise


# Example usage
# storage = IndexStorage()
# storage.persist_index(index, "my_index", "/path/to/indexing_directory")
# storage.log_persist_action(index, "my_index", "/path/to/indexing_directory")
# loaded_index = IndexStorage.load_index("my_index", "my_storage")
