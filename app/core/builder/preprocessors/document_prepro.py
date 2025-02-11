from typing import List
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from app.logging_config import indexing_logger as logger
# Define the expected structured output using a Pydantic model.
class DocumentFeatures(BaseModel):
    summary: str = Field(description="A comprehensive summary of the document in 10-15 lines covering its core content and important sections.")
    highlights: List[str] = Field(description="Five key highlights from the document.")
    document_type: str = Field(description="A one-word descriptor indicating the type of document.")
    
prompt_template = """

    You are an expert summarizer. Using the combined summaries below,
    please produce a structured output in valid JSON format that follows this schema:
    
    1. "summary": "A comprehensive summary of the document in 10-15 lines covering its core content and important sections.",
    2. "highlights": ["Highlight 1", "Highlight 2", "Highlight 3", "Highlight 4", "Highlight 5"],
    3. "document_type": "A one-word descriptor indicating the type of document."
       
    Combined Summaries:
    {combined_text}
    
    """

prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["combined_text"]
    )

def generate_document_features(summaries: List[Document], llm: any) -> DocumentFeatures:
    """
    Combines a list of summary chunks, sends them to the LLM with structured output instructions,
    and returns a DocumentSummary object containing:
      - a comprehensive 10–15 line summary,
      - five key highlights,
      - and a one-word document type descriptor.
    
    Parameters:
        summaries (List[str]): List of summary strings.
        llm (OpenAI): An instantiated LLM (or any compatible model) from LangChain.
    
    Returns:
        DocumentSummary: A structured output object with the summary, highlights, and document type.
    """
    # Step 1: Combine the input summaries.
    combined_text = "\n\n".join([doc.page_content for doc in summaries])
    
    model = llm.with_structured_output(DocumentFeatures)
    formatted_prompt = prompt.format(combined_text=combined_text)
    response = model.invoke(formatted_prompt)
    logger.info(f"Structured output response by generate document features: {response}")
    return response
