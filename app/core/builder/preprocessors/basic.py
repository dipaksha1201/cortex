
class BasicPreprocessor:
    @staticmethod
    def split_docs_by_separator(docs, separator="\n---\n"):
            """Split docs into sub-documents based on a separator."""
            try:
                sub_docs = []
                for doc in docs:
                    doc_chunks = doc.text.split(separator)
                    for doc_chunk in doc_chunks:
                        sub_doc = Document(
                            text=doc_chunk,
                            metadata=doc.metadata,
                        )
                        sub_docs.append(sub_doc)

                return sub_docs
            except Exception as e:
                logger.error(f"Error splitting documents: {e}")
                raise