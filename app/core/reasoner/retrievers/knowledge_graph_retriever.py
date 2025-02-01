import logging
from llama_index.core.indices.property_graph import (
    PGRetriever,
    VectorContextRetriever,
    LLMSynonymRetriever,
)

from app.core.builder.indexer.knowledge_graph_indexer import KnowledgeGraphIndexer

# Configure logging
logger = logging.getLogger(__name__)

class KnowledgeGraphRetriever:
    """Retriever class for knowledge graph based retrieval"""
    
    def __init__(self, index_name):
        """Initialize the retriever with a knowledge graph index"""
        kg_index = KnowledgeGraphIndexer()
        self.index = kg_index.get_index_from_storage(index_name)
        
    def _process_source_nodes(self, source_nodes):
        """
        Process source nodes and concatenate their text content
        
        Args:
            source_nodes (List[NodeWithScore]): List of retrieved nodes with scores
            
        Returns:
            str: Concatenated text from all source nodes
        """
        combined_text = []
        for node in source_nodes:
            logger.debug(f"Processing node: {node}")
            if hasattr(node, 'text'):
                combined_text.append(node.text)
            else:
                logger.debug("Node does not have text attribute")
            
        concatenated_text = " ".join(combined_text)
        logger.info(f"Concatenated text: {concatenated_text}")
        return concatenated_text
        
    def retrieve(self, query, max_results=5):
        """
        Retrieve relevant information from the knowledge graph based on the query
        
        Args:
            query (str): The query string to search for
            max_results (int): Maximum number of results to return (default: 5)
            
        Returns:
            List of relevant nodes/documents from the knowledge graph
        """
        try:
            logger.debug(f"Retrieving results for query: {query}")
            sub_retrievers = [
                VectorContextRetriever(self.index.property_graph_store, ...),
                LLMSynonymRetriever(self.index.property_graph_store, ...),
            ]

            retriever = PGRetriever(sub_retrievers=sub_retrievers)
            source_nodes = retriever.retrieve(query)
            
            # Get source nodes from response
            if len(source_nodes) > max_results:
                source_nodes = source_nodes[:max_results]
                
            logger.info(f"Retrieved {len(source_nodes)} results for query")
            combined_text = self._process_source_nodes(source_nodes)
            return combined_text
            
        except Exception as e:
            logger.error(f"Error retrieving results: {e}")
            raise


