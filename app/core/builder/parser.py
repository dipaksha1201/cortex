from llama_parse import LlamaParse
from dotenv import load_dotenv
import os

load_dotenv()

class Parser:
    """
    A static class for parsing documents and queries.
    """
    @staticmethod
    def initialize_parser():
        return LlamaParse(
            api_key=os.getenv("LLAMAPARSER_API_KEY"),  # Use API key from .env
            result_type="markdown",  # "markdown" and "text" are available
            verbose=True,
            language="en",  # Optionally you can define a language, default=en
        )

    @staticmethod
    def load_documents(file_path):
        parser = Parser.initialize_parser()
        documents = parser.load_data(file_path)
        return documents

    def __new__(cls):
        raise NotImplementedError("Cannot instantiate a static class")

# Example usage
# documents = Parser.load_documents("../data/kag_paper.pdf")
# Parser.store_parsed_document(parsed_data, "parsed_document", "./parsed_docs")
