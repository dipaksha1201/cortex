from ....initialization import gemini_pro_model
from ...interface.base_indexer import BaseIndexer
from ...reasoner.retriever import MultiVectorRetrieverBuilder
from ..preprocessors.multivector_langchain import MultiVectorLangchain
import logging
from uuid import uuid4

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreIndexer(BaseIndexer):
    
    def __init__(self):
        self.retriever_builder = MultiVectorRetrieverBuilder()
        self.model = gemini_pro_model  # Import this from your initialization module

    def index(self, file_name, index_name, documents):
        """Index documents using MultiVectorRetriever"""
        try:
            
            # Generate unique IDs for documents
            doc_ids = [str(uuid4()) for _ in documents]
            
            # Initialize MultiVectorLangchain processor
            processor = MultiVectorLangchain(
                documents=documents,
                doc_ids=doc_ids,
                file_name=file_name,
                model=self.model,  # You'll need to initialize this in __init__
                id_key="doc_id"
            )
            
            # Process documents to get chunks, summaries, and questions
            document_chunks, summary_docs, question_docs = processor.process_documents()
            
            # Combine all document types
            combined_docs = document_chunks + summary_docs + question_docs
            
            # Get retriever for the file
            retriever = self.retriever_builder.build(index_name)
            
            # Add documents to vector store
            retriever.vectorstore.add_documents(combined_docs)
            
            # Store original documents in docstore
            langchain_docs = processor.convert_to_langchain_docs()
            retriever.docstore.mset(list(zip(doc_ids, langchain_docs)))
            
            logger.info(f"Successfully indexed documents for {index_name}")
            return retriever
            
        except Exception as e:
            logger.error(f"Error in indexing: {e}")
            raise

    def get_index_from_storage(self, index_name):
        """Get existing retriever from storage"""
        try:
            # Create retriever with existing index
            retriever = self.retriever_builder.build(index_name)
            return retriever
        except Exception as e:
            logger.error(f"Error loading index from storage: {e}")
            raise