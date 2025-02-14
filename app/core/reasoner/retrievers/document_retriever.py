from typing import List
from langchain_core.documents import Document
from app.logging_config import retriever_logger as logger
from ...common.multivector_retriever import MultiVectorRetrieverBuilder

def retrieve_summaries_from_store(index_name: str, target_filename: str, column_name: str) -> List[Document]:
    """
    Retrieves summary documents for a given filename from the vector store.
    
    Parameters:
        index_name (str): Name of the index to search in
        target_filename (str): The filename to filter by
        
    Returns:
        List[Document]: List of summary documents matching the filename
    """
    try:
        retriever_builder = MultiVectorRetrieverBuilder()
        retriever = retriever_builder.build(index_name)
        
        # Query the vector store for summaries
        # Using a metadata filter for the source filename
        summaries = retriever.vectorstore.similarity_search(
            column_name,  # Using "summary" as query since we want summary documents
            filter={
                "source": {"$eq": target_filename},
                "document_type": {"$eq": "summary"}
            },
            k=100  # Adjust this number based on expected number of summaries
        )
        
        logger.info(f"Retrieved {len(summaries)} summaries for file {target_filename}")
        return summaries
        
    except Exception as e:
        logger.error(f"Error retrieving summaries for {target_filename}: {str(e)}")
        raise