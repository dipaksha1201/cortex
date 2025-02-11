from app.data_layer.models.document import Document
from app.data_layer.services.document_service import DocumentService
from .parser import Parser
from .indexer import KnowledgeGraphIndexer, VectorStoreIndexer
from app.logging_config import indexing_logger as logger

class Indexer:
    def __init__(self):
        logger.info("Initializing Indexer")
        self.knowledge_graph_indexer = KnowledgeGraphIndexer()
        self.vector_store_indexer = VectorStoreIndexer()
        # self.analytical_indexer = AnalyticalIndexer()

    async def index(self, file, index_name):
        logger.info(f"Starting indexing process for file with index name: {index_name}")
        try:
            logger.info(f"Attempting to parse file using Parser.load_documents")
            file_name = file.filename    
            parsed_results = await Parser.load_data(file)
            
            if "parsed_content" not in parsed_results:
                logger.error(f"Error parsing file: {parsed_results['error']}")
                return False
            
            documents = parsed_results.get("parsed_content")
            logger.info(f"Successfully parsed documents from file: {file_name}")

            logger.info(f"Indexing documents in knowledge graph with index: {index_name}")
            kg_status = self.knowledge_graph_indexer.index(index_name, documents)

            logger.info(f"Indexing documents in vector store with file: {file_name}, index: {index_name}")
            vector_status , document_features = self.vector_store_indexer.index(file_name, index_name, documents)
            
            document = Document(
                user_id=index_name,
                name=file_name,
                type=document_features.document_type,
                summary=document_features.summary,
                highlights=document_features.highlights
            )
            
            service = DocumentService()
            service.insert_document(document)
            
            if kg_status and vector_status:
                logger.info(f"Successfully indexed file {file_name}")
                return True
            else:
                logger.error(f"Failed to index file {file_name}")
                return False

        except Exception as e:
            logger.error(f"Error indexing file: {str(e)}", exc_info=True)
            return False
