import logging
from llama_index.core import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BasicPreprocessor:
    @staticmethod
    def split_docs_by_separator(parsed_results, separator="\n---\n"):
            """Split docs into sub-documents based on a separator."""
            try:
                sub_docs = []
                doc_chunks = parsed_results.split(separator)
                for doc_chunk in doc_chunks:
                    sub_doc = Document(
                        text=doc_chunk,
                        metadata={}
                    )
                    sub_docs.append(sub_doc)

                return sub_docs
            except Exception as e:
                logger.error(f"Error splitting documents: {e}")
                raise