from ..preprocessors import BasicPreprocessor
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import Document, SummaryIndex
from app.storage import DiskStore
from ...interface.base_indexer import BaseIndexer

index_source = "sparse"
class SparseIndexer(BaseIndexer):
    def __init__(self):
        pass

    def index(self, index_name, documents=None):

        if documents is None:
            index = self.get_index_from_storage(index_name)
            return index

        document_chunks = self.create_chunks_from_documents(documents)
        summary_index = SummaryIndex.from_documents(document_chunks)
        DiskStore.persist_index(summary_index, index_source, index_name)
        return summary_index

    def get_index_from_storage(self, index_name):
        loaded_index = DiskStore.load_index(index_source,index_name)
        return loaded_index

    def convert_subdocs_to_documents(self, sub_docs):
        # Initialize the parser with desired chunk size and overlap
        parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=50)
        # Get text nodes from the sub-documents
        text_nodes = parser.get_nodes_from_documents(sub_docs)
        # Create Document objects from text chunks
        documents = [Document(text=chunk.text) for chunk in text_nodes]
        return documents

    def create_chunks_from_documents(self, documents):
        sub_docs = BasicPreprocessor.split_docs_by_separator(documents)
        return self.convert_subdocs_to_documents(sub_docs)

# Example usage
# analytical_indexer = AnalyticalIndexer()
# documents = analytical_indexer.create_documents_from_subdocs(sub_docs)
