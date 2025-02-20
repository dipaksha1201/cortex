from app.data_layer.models.document import Document
from app.data_layer.services.document_service import DocumentService
from .parser import Parser
from .indexer import KnowledgeGraphIndexer, VectorStoreIndexer
from app.logging_config import indexing_logger as logger
from app.core.builder.indexer.sparse_indexer import SparseIndexer
import asyncio
from typing import Dict, Any, Tuple, Union

class Indexer:
    def __init__(self):
        logger.info("Initializing Indexer")
        self.knowledge_graph_indexer = KnowledgeGraphIndexer()
        self.vector_store_indexer = VectorStoreIndexer()
        self.sparse_indexer = SparseIndexer()
        # self.analytical_indexer = AnalyticalIndexer()

    async def _run_kg_indexer(self, index_name, documents) -> Tuple[str, Union[bool, Exception]]:
        logger.info(f"Running knowledge graph indexer with index: {index_name}")
        try:
            result = await asyncio.to_thread(self.knowledge_graph_indexer.index, index_name, documents)
            return "kg", result
        except Exception as e:
            logger.error(f"Error in knowledge graph indexer: {str(e)}")
            return "kg", e

    async def _run_sparse_indexer(self, index_name, documents) -> Tuple[str, Union[bool, Exception]]:
        logger.info(f"Running sparse indexer with index: {index_name}")
        try:
            result = await asyncio.to_thread(self.sparse_indexer.index, index_name, documents)
            return "sparse", result
        except Exception as e:
            logger.error(f"Error in sparse indexer: {str(e)}")
            return "sparse", e

    async def _run_vector_indexer(self, file_name, index_name, documents) -> Tuple[str, Union[Tuple[bool, Any], Exception]]:
        logger.info(f"Running vector store indexer with file: {file_name}, index: {index_name}")
        try:
            result = await asyncio.to_thread(self.vector_store_indexer.index, file_name, index_name, documents)
            return "vector", result
        except Exception as e:
            logger.error(f"Error in vector store indexer: {str(e)}")
            return "vector", e

    async def index(self, file, index_name):
        logger.info(f"Starting indexing process for file with index name: {index_name}")
        max_attempts = 2  # Number of attempts to try indexing
        attempt = 0
        failed_indices = []

        while attempt < max_attempts:
            try:
                logger.info(f"Attempt {attempt + 1}: Attempting to parse file using Parser.load_documents")
                file_name = file.filename    
                parsed_results = await Parser.load_data(file)
                
                if "parsed_content" not in parsed_results:
                    logger.error(f"Error parsing file: {parsed_results['error']}")
                    return False
                
                documents = parsed_results.get("parsed_content")
                logger.info(f"Successfully parsed documents from file: {file_name}")

                # Initialize results tracking
                indexer_results = {}
                document_features = None

                # Run KG indexer
                if attempt == 0 or "kg" in failed_indices:
                    # kg_type, kg_result = await self._run_kg_indexer(index_name, documents)
                    kg_type, kg_result = self.knowledge_graph_indexer.index(index_name, documents)
                    if isinstance(kg_result, Exception) or kg_result is False:
                        failed_indices.append(kg_type)
                        indexer_results[kg_type] = False
                        logger.error(f"KG indexer failed on attempt {attempt + 1}")
                    else:
                        indexer_results[kg_type] = kg_result
                        if kg_type in failed_indices:
                            failed_indices.remove(kg_type)
                        logger.info("KG indexer succeeded")

                # Run vector indexer
                if attempt == 0 or "vector" in failed_indices:
                    # vector_type, vector_result = await self._run_vector_indexer(file_name, index_name, documents)
                    vector_type, vector_result = self.vector_store_indexer.index(file_name, index_name, documents)
                    if isinstance(vector_result, Exception) or vector_result is False:
                        failed_indices.append(vector_type)
                        indexer_results[vector_type] = False
                        logger.error(f"Vector indexer failed on attempt {attempt + 1}")
                    else:
                        vector_status, doc_features = vector_result
                        indexer_results[vector_type] = vector_status
                        if vector_status:
                            document_features = doc_features
                            if vector_type in failed_indices:
                                failed_indices.remove(vector_type)
                            logger.info("Vector indexer succeeded")
                        else:
                            logger.error(f"Vector indexer returned false status on attempt {attempt + 1}")
                            failed_indices.append(vector_type)

                # Check if we need to retry any failed indexers
                if failed_indices:
                    logger.error(f"Failed indexers: {', '.join(failed_indices)}")
                    attempt += 1
                    if attempt < max_attempts:
                        logger.info("Retrying failed indexers...")
                        continue
                    return False

                # If we get here, both indexers succeeded
                document = Document(
                    user_id=index_name,
                    name=file_name,
                    type=document_features.document_type,
                    summary=document_features.summary,
                    highlights=document_features.highlights
                )
                
                service = DocumentService()
                service.insert_document(document)
                logger.info("Successfully completed indexing process")
                return True

            except Exception as e:     
                logger.error(f"Exception during indexing attempt {attempt + 1}: {str(e)}")
                attempt += 1
                if attempt < max_attempts:
                    logger.info("Retrying indexing process...")
                    continue
                return False
