import logging
from typing import List, Tuple
from llama_index.core import SummaryIndex
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.retrievers.bm25.base import BM25Retriever
from app.storage import DiskStore
from app.logging_config import retriever_logger as logger
from llama_index.core.retrievers import VectorIndexRetriever
from concurrent.futures import ThreadPoolExecutor

class SparseRetriever:
    """Hybrid retriever combining vector and keyword-based retrieval"""
    
    def __init__(self, index_name: str):
        """Initialize with loaded sparse index"""
        try:
            logger.info(f"Initializing SparseRetriever for index '{index_name}'")
            self.index = DiskStore.load_index("sparse", index_name)
            
            # Initialize keyword retriever
            self.retriever = self.index.as_retriever(
                retriever_mode="keyword",
                similarity_top_k=5
            )
            
            # Initialize vector retriever
            self.vector_retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=5
            )
            
            # Create recursive retriever
            self.recursive_retriever = RecursiveRetriever(
                "vector",
                retriever_dict={
                    "vector": self.vector_retriever,
                    "keyword": self.retriever
                },
                verbose=True
            )
            
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=5
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize SparseRetriever: {e}")
            raise

    def retrieve(self, query: str, score_threshold: float = 0.45) -> Tuple[List, str]:
        """Hybrid retrieval with batched processing"""
        try:
            logger.info(f"Executing hybrid retrieval for: {query}")
            
            # Get vector results
            vector_nodes = self.vector_retriever.retrieve(query)
            
            # Get keyword results
            keyword_nodes = self.retriever.retrieve(query)
            
            # Combine and filter
            all_nodes = vector_nodes + keyword_nodes
            logger.info(f"All nodes: {all_nodes}")
            filtered_nodes = [
                n for n in all_nodes
                if hasattr(n, 'score') and n.score >= score_threshold
            ]
            
            # Parallel processing of filtered nodes
            batch_size = 10
            with ThreadPoolExecutor() as executor:
                batches = [filtered_nodes[i:i+batch_size] 
                          for i in range(0, len(filtered_nodes), batch_size)]
                
                processed_batches = list(executor.map(
                    lambda batch: self._process_batch(batch),
                    batches
                ))
            
            unique_nodes = {}
            for batch in processed_batches:
                unique_nodes.update(batch)
            
            combined_text = " ".join(unique_nodes.values())
            
            logger.info(f"Retrieved {len(unique_nodes)} relevant nodes")
            return list(unique_nodes.keys()), combined_text
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            raise

    def _process_batch(self, batch: List) -> dict:
        """Process a batch of nodes"""
        batch_nodes = {}
        for n in batch:
            try:
                if n.node.id_ not in batch_nodes:
                    batch_nodes[n.node.id_] = n.node.text
            except AttributeError:
                continue
        return batch_nodes

    def _process_nodes(self, nodes: List) -> str:
        """Process and combine node contents"""
        return " ".join(
            node.text for node in nodes 
            if hasattr(node, 'text') and node.text.strip()
        )

    def query_index(self, query: str):
        """Streaming response for real-time applications"""
        try:
            query_index = self.query_engine.query(query)
            return query_index
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield "Error processing request"

    def stream_response(self, query: str):
        """Streaming response for real-time applications"""
        try:
            streaming_response = self.query_engine.query(query)
            for text in streaming_response.response_gen:
                yield f"data: {text}\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield "data: Error processing request\n\n" 