import logging
from typing import List, Tuple
from collections import defaultdict

from app.core.common.multivector_retriever import MultiVectorRetrieverBuilder


logger = logging.getLogger(__name__)

class VectorRetriever:
    """Retriever class for vector-based retrieval"""
    
    def __init__(self, index_name: str):
        """Initialize the retriever with a vector index
        
        Args:
            index_name (str): Name of the vector index to use
        """
        multi_vector_retriever = MultiVectorRetrieverBuilder()
        self.retriever = multi_vector_retriever.build(index_name)
        
    def retrieve_with_threshold(self,search_results: List,score_threshold: float = 0.7) -> Tuple[List, str]:
        """
        Retrieve documents that meet a minimum similarity score threshold
        
        Args:
            query (str): The query string to search for
            score_threshold (float): Minimum similarity score to include a document (default: 0.7)
            max_results (int): Maximum number of results to return (default: 5)
            
        Returns:
            tuple: (List of filtered documents, Combined text string)
        """
        try:
            id_to_doc = defaultdict(list)
            for doc, score in search_results:
                if score >= score_threshold:
                    doc_id = doc.metadata.get("doc_id")
                    if doc_id:
                        doc.metadata["score"] = score
                        id_to_doc[doc_id].append(doc)
            
            # Fetch parent documents and attach sub-documents
            filtered_docs = []
            for doc_id, sub_docs in id_to_doc.items():
                docstore_docs = self.retriever.docstore.mget([doc_id])
                if docstore_docs and (doc := docstore_docs[0]):
                    doc.metadata["sub_docs"] = sub_docs
                    filtered_docs.append(doc)
            
            return filtered_docs
            
        except Exception as e:
            logger.error(f"Error retrieving results with threshold: {e}")
            raise
    
    def retrieve(self, query: str, max_results: int = 3) -> Tuple[List, str]:
        """
        Retrieve relevant information from the vector store based on the query
        
        Args:
            query (str): The query string to search for
            max_results (int): Maximum number of results to return (default: 5)
            
        Returns:
            tuple: (List of source nodes, Combined text string)
        """
        try:
            logger.debug(f"Retrieving results for query: {query}")
            source_nodes = self.retriever.vectorstore.similarity_search_with_score(query , k = max_results)
            filtered_docs = self.retrieve_with_threshold(source_nodes)
            combined_text = self._process_source_nodes(filtered_docs)
            logger.info(f"Retrieved {len(filtered_docs)} results for query")
            return combined_text
            
        except Exception as e:
            logger.error(f"Error retrieving results: {e}")
            raise
            
    def _process_source_nodes(self, source_nodes: List) -> str:
        """
        Process source nodes and concatenate their text content
        
        Args:
            source_nodes (List): List of retrieved nodes
            
        Returns:
            str: Concatenated text from all source nodes
        """
        combined_text = []
        for node in source_nodes:
            if hasattr(node, 'text'):
                combined_text.append(node.text)
            elif hasattr(node, 'page_content'):  # For LangChain Document objects
                combined_text.append(node.page_content)
            
        return " ".join(combined_text)
