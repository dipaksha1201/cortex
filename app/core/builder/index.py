from .parser import Parser
from .indexer import KnowledgeGraphIndexer, VectorStoreIndexer


class Indexer:
    def __init__(self):
        self.knowledge_graph_indexer = KnowledgeGraphIndexer()
        self.vector_store_indexer = VectorStoreIndexer()
        # self.analytical_indexer = AnalyticalIndexer()

    def index(self , file, index_name):
        documents = Parser.load_documents(file)
        file_name = file.filename if hasattr(file, 'filename') else file.name
        self.knowledge_graph_indexer.index(index_name, documents)
        self.vector_store_indexer.index(file_name ,index_name, documents)
        # self.analytical_indexer.index(index_name, documents)
        return True

# Example usage
# indexer = Indexer()
# indexer.knowledge_graph_indexing(data)
# indexer.vector_store_indexing(data)
# indexer.analytical_indexing(data)
