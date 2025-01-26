from typing import Any, Dict
from .builder import Parser, Preprocessor, Indexer
from .reasoner import Retriever

class Agent:
    def handle_parsing(self, query: str, parser: Parser) -> Dict[str, Any]:
        """
        Handle the parsing of the input query
        """
        return parser.parse(query)

    def handle_preprocessing(self, data: Dict[str, Any], preprocessor: Preprocessor) -> Dict[str, Any]:
        """
        Handle the preprocessing of parsed data
        """
        return preprocessor.process(data)

    def handle_indexing(self, data: Dict[str, Any], indexer: Indexer) -> Dict[str, Any]:
        """
        Handle the indexing of preprocessed data
        """
        return indexer.index(data)

    def handle_retrieval(self, data: Dict[str, Any], retriever: Retriever) -> Dict[str, Any]:
        """
        Handle the retrieval based on indexed data
        """
        return retriever.retrieve(data)
