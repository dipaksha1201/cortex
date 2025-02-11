from app.core.builder.preprocessors.document_prepro import generate_document_features
from ....initialization import gemini_pro_model_langchain, gemini_flash_model_langchain
from ...interface.base_indexer import BaseIndexer
from ...common.multivector_retriever import MultiVectorRetrieverBuilder
from ..preprocessors.multivector_langchain import MultiVectorLangchain
import logging
from uuid import uuid4

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreIndexer(BaseIndexer):
    
    def __init__(self):
        self.retriever_builder = MultiVectorRetrieverBuilder()
        self.model = gemini_pro_model_langchain  # Import this from your initialization module

    def index(self, file_name, index_name, documents):
        """Index documents using MultiVectorRetriever"""
        logger.debug(f"Starting vector store indexing for file '{file_name}' with index '{index_name}'")
        try:
            # Generate unique IDs for documents
            logger.debug(f"Generating unique IDs for {len(documents)} documents")
            doc_ids = [str(uuid4()) for _ in documents]
            
            # Initialize MultiVectorLangchain processor
            logger.debug("Initializing MultiVectorLangchain processor")
            processor = MultiVectorLangchain(
                documents=documents,
                doc_ids=doc_ids,
                file_name=file_name,
                model=self.model,
                id_key="doc_id"
            )
            retriever = self.retriever_builder.build(index_name)
            
            # Process documents to get chunks, summaries, and questions
            logger.debug("Processing documents to generate chunks, summaries and questions")
            document_chunks, summary_docs, question_docs = processor.process_documents()
            logger.debug(f"Generated {len(document_chunks)} chunks, {len(summary_docs)} summaries, {len(question_docs)} questions")
            
            
            # Combine all document types
            logger.debug("Combining all document types")
            combined_docs = document_chunks + summary_docs + question_docs
            
            # Get retriever for the file
            logger.debug(f"Building retriever for index '{index_name}'")
            
            # Add documents to vector store
            logger.debug(f"Adding {len(combined_docs)} documents to vector store")
            retriever.vectorstore.add_documents(combined_docs)
            
            # Store original documents in docstore
            logger.debug("Converting and storing original documents in docstore")
            langchain_docs = processor.convert_to_langchain_docs()
            retriever.docstore.mset(list(zip(doc_ids, langchain_docs)))
            
            document = generate_document_features(summary_docs, gemini_flash_model_langchain)
            
            logger.info(f"Successfully indexed {len(langchain_docs)} documents for '{index_name}'")
            return True , document
            
        except Exception as e:
            logger.error(f"Error indexing documents for '{index_name}': {str(e)}", exc_info=True)
            return False

    def get_index_from_storage(self, index_name):
        """Get existing retriever from storage"""
        try:
            # Create retriever with existing index
            retriever = self.retriever_builder.build(index_name)
            return retriever
        except Exception as e:
            logger.error(f"Error loading index from storage: {e}")
            raise