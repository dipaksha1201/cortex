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
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.sparse_indexer.index, index_name, documents)
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
        try:
            logger.info(f"Attempting to parse file using Parser.load_documents")
            file_name = file.filename    
            parsed_results = await Parser.load_data(file)
            
            if "parsed_content" not in parsed_results:
                logger.error(f"Error parsing file: {parsed_results['error']}")
                return False
            
            documents = parsed_results.get("parsed_content")
            logger.info(f"Successfully parsed documents from file: {file_name}")

            # Run all indexers in parallel
            tasks = [
                self._run_kg_indexer(index_name, documents),
                # self._run_sparse_indexer(index_name, documents),
                self._run_vector_indexer(file_name, index_name, documents)
            ]
            
            # Wait for all tasks to complete, collecting results and errors
            results = await asyncio.gather(*tasks, return_exceptions=False)
            
            # Process results
            indexer_results = {}
            document_features = None
            failed_indices = []
            
            for indexer_type, result in results:
                if isinstance(result, Exception):
                    failed_indices.append(f"{indexer_type} ({str(result)})")
                    indexer_results[indexer_type] = False
                else:
                    if indexer_type == "vector":
                        vector_status, doc_features = result
                        indexer_results[indexer_type] = vector_status
                        if vector_status:
                            document_features = doc_features
                    else:
                        indexer_results[indexer_type] = result

            # Only create document if vector indexing succeeded
            if document_features and indexer_results.get("vector"):
                document = Document(
                    user_id=index_name,
                    name=file_name,
                    type=document_features.document_type,
                    summary=document_features.summary,
                    highlights=document_features.highlights
                )
                
                service = DocumentService()
                service.insert_document(document)
            
            # Check if any indexers succeeded
            if any(indexer_results.values()):
                succeeded = [k for k, v in indexer_results.items() if v]
                logger.info(f"Successfully indexed in: {', '.join(succeeded)}")
                
                if failed_indices:
                    logger.error(f"Failed indexers: {', '.join(failed_indices)}")
                
                # Return true if at least vector store succeeded (since we need document features)
                return indexer_results.get("vector", False)
            else:
                logger.error(f"All indexers failed: {', '.join(failed_indices)}")
                return False

        except Exception as e:
            logger.error(f"Error during indexing process: {str(e)}", exc_info=True)
            return False
