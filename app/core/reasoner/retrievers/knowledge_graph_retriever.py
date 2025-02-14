import logging
from llama_index.core.indices.property_graph import (
    PGRetriever,
    VectorContextRetriever,
    LLMSynonymRetriever,
)
import asyncio
from app.core.builder.indexer.knowledge_graph_indexer import KnowledgeGraphIndexer

# Configure logging
logger = logging.getLogger(__name__)

class KnowledgeGraphRetrieverError(Exception):
    """Custom exception for KnowledgeGraphRetriever errors"""
    pass

class KnowledgeGraphRetriever:
    """Retriever class for knowledge graph based retrieval"""
    
    def __init__(self, index_name):
        """Initialize the retriever with a knowledge graph index"""
        try:
            kg_index = KnowledgeGraphIndexer()
            self.index = kg_index.get_index_from_storage(index_name)
            if not self.index:
                raise KnowledgeGraphRetrieverError(f"Failed to load index for {index_name}")
        except Exception as e:
            logger.error(f"Error initializing KnowledgeGraphRetriever: {e}")
            raise KnowledgeGraphRetrieverError(f"Failed to initialize retriever: {str(e)}")
        
    def _process_source_nodes(self, source_nodes):
        """
        Process source nodes and concatenate their text content
        
        Args:
            source_nodes (List[NodeWithScore]): List of retrieved nodes with scores
            
        Returns:
            str: Concatenated text from all source nodes
        """
        if not source_nodes:
            logger.warning("No source nodes to process")
            return ""
            
        try:
            combined_text = []
            for node in source_nodes:
                logger.debug(f"Processing node: {node}")
                if hasattr(node, 'text'):
                    combined_text.append(node.text)
                else:
                    logger.debug(f"Node {node} does not have text attribute")
            
            concatenated_text = " ".join(combined_text)
            logger.info(f"Concatenated text: {concatenated_text}")
            return concatenated_text
        except Exception as e:
            logger.error(f"Error processing source nodes: {e}")
            raise KnowledgeGraphRetrieverError(f"Failed to process source nodes: {str(e)}")
        
    async def aretrieve(self, query, max_results=5):
        """Async retrieve method for knowledge graph"""
        if not query:
            logger.error("Empty query provided")
            raise ValueError("Query cannot be empty")
            
        try:
            logger.info(f"Retrieving results for query from KG retriever: {query}")
            
            if not self.index or not self.index.property_graph_store:
                raise KnowledgeGraphRetrieverError("Index or property graph store not initialized")
                
            sub_retrievers = [
                VectorContextRetriever(self.index.property_graph_store, ...),
                LLMSynonymRetriever(self.index.property_graph_store, ...),
            ]

            retriever = PGRetriever(sub_retrievers=sub_retrievers)
            source_nodes = await retriever.aretrieve(query)
            
            if not source_nodes:
                logger.warning(f"No results found for query: {query}")
                return ""
                
            if len(source_nodes) > max_results:
                source_nodes = source_nodes[:max_results]
                
            logger.info(f"Retrieved {len(source_nodes)} results for query")
            return self._process_source_nodes(source_nodes)
            
        except asyncio.CancelledError:
            logger.error("Async retrieval was cancelled")
            raise
        except Exception as e:
            logger.error(f"Error retrieving results: {e}")
            raise KnowledgeGraphRetrieverError(f"Failed to retrieve results: {str(e)}")
            
    def retrieve(self, query, max_results=5):
        """Synchronous wrapper for aretrieve"""
        if not query:
            logger.error("Empty query provided")
            raise ValueError("Query cannot be empty")
            
        try:
            return asyncio.run(self.aretrieve(query, max_results))
        except asyncio.CancelledError:
            logger.error("Retrieval operation was cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in sync retrieve: {e}")
            raise KnowledgeGraphRetrieverError(f"Failed to retrieve results synchronously: {str(e)}")


