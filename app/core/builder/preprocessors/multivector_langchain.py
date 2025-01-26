from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from pydantic import BaseModel, Field

class MultiVectorLangchain:
    def __init__(self, documents, doc_ids, filepath, gemini_llm, id_key):
        self.documents = documents
        self.doc_ids = doc_ids
        self.filepath = filepath
        self.gemini_llm = gemini_llm
        self.id_key = id_key

    def convert_to_langchain_docs(self):
        """Convert parsed documents to LangChain Document format."""
        langchain_docs = [
            Document(page_content=doc.text, metadata={"source": self.filepath})
            for doc in self.documents
        ]
        return langchain_docs

    def split_into_smaller_chunks(self, langchain_docs):
        """Split LangChain documents into smaller chunks."""
        child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
        document_chunks = []
        for i, doc in enumerate(langchain_docs):
            _id = self.doc_ids[i]
            _sub_docs = child_text_splitter.split_documents([doc])
            for _doc in _sub_docs:
                _doc.metadata[self.id_key] = _id
            document_chunks.extend(_sub_docs)
        return document_chunks

    def generate_summaries(self, langchain_docs):
        """Generate summaries for the LangChain documents."""
        chain = (
            {"doc": lambda x: x.page_content}
            | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
            | self.gemini_llm
            | StrOutputParser()
        )
        summaries = chain.batch(langchain_docs, {"max_concurrency": 5})
        summary_docs = [
            Document(page_content=s, metadata={self.id_key: self.doc_ids[i],})
            for i, s in enumerate(summaries)
        ]
        return summary_docs

    def generate_hypothetical_questions(self, langchain_docs):
        """Generate hypothetical questions based on the LangChain documents."""
        class HypotheticalQuestions(BaseModel):
            questions: List[str] = Field(..., description="List of questions")

        chain = (
            {"doc": lambda x: x.page_content}
            | ChatPromptTemplate.from_template(
                "Generate a list of exactly 3 hypothetical questions that the below document could be used to answer:\n\n{doc}"
            )
            | self.gemini_llm.with_structured_output(HypotheticalQuestions)
            | (lambda x: x.questions)
        )

        hypothetical_questions = chain.batch(langchain_docs, {"max_concurrency": 5})
        question_docs = []
        for i, question_list in enumerate(hypothetical_questions):
            question_docs.extend(
                [Document(page_content=s, metadata={self.id_key: self.doc_ids[i]}) for s in question_list]
            )

        return question_docs

    def process_documents(self):
        """Process the documents by calling the other methods."""
        langchain_docs = self.convert_to_langchain_docs()
        document_chunks = self.split_into_smaller_chunks(langchain_docs)
        summary_docs = self.generate_summaries(langchain_docs)
        question_docs = self.generate_hypothetical_questions(langchain_docs)
        return document_chunks, summary_docs, question_docs

# Example usage
# mv_langchain = MultiVectorLangchain(documents, doc_ids, filepath, gemini_llm, id_key)
# document_chunks, summary_docs, question_docs = mv_langchain.process_documents()