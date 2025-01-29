from abc import ABC, abstractmethod

class BaseIndexer(ABC):
    """Base interface for all indexers"""
    
    @abstractmethod
    def index(self, file_name, documents=None):
        """
        Index the given documents or load existing index
        
        Args:
            file_name: Name of the file/index to save/load
            documents: Documents to index. If None, loads existing index
            
        Returns:
            The created or loaded index
        """
        pass

    @abstractmethod
    def get_index_from_storage(self, file_name):
        """
        Load an existing index from storage
        
        Args:
            file_name: Name of the index file to load
            
        Returns:
            The loaded index
        """
        pass
