from app.core.common.multivector_retriever import MultiVectorRetrieverBuilder
from app.core.reasoner.retrievers.document_retriever import retrieve_summaries_from_store
from app.core.builder.preprocessors.document_prepro import retrieve_column_information
from app.initialization import gemini_flash_model_langchain
from typing import Union, List
from app.logging_config import service_logger as logger

class Documents:
    def __init__(self):
        self.retriever = MultiVectorRetrieverBuilder()
        logger.info("Documents service initialized")

    def get_document_column_info(self, user_id: str, file_name: str, column_name: str) -> Union[str, List[str]]:
        """
        Retrieves and extracts specific column information for a given file.
        
        Parameters:
            user_id (str): The ID of the user
            file_name (str): Name of the file to get information for
            column_name (str): Name of the column to extract (e.g., 'summary', 'highlights', 'document_type')
            
        Returns:
            Union[str, List[str]]: The extracted column value, either a string or list of strings
        """
        logger.info(f"Retrieving {column_name} for file '{file_name}' (user: {user_id})")
        
        try:
            summaries = retrieve_summaries_from_store(user_id, file_name, column_name)
            
            if not summaries:
                logger.warning(f"No summaries found for file '{file_name}' (user: {user_id})")
                default_value = [] if column_name == "highlights" else ""
                logger.info(f"Returning default value for {column_name}: {default_value}")
                return default_value
            
            logger.info(f"Summaries retrieved: {summaries}")
            column_info = retrieve_column_information(summaries, column_name, gemini_flash_model_langchain)
            logger.info(f"Successfully retrieved {column_name} for file '{file_name}' (user: {user_id})")
            return column_info
            
        except Exception as e:
            logger.error(
                f"Error retrieving {column_name} for file '{file_name}' (user: {user_id})",
                exc_info=True,
                extra={
                    "user_id": user_id,
                    "file_name": file_name,
                    "column_name": column_name,
                    "error": str(e)
                }
            )
            raise

