from langchain.output_parsers import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pydantic import BaseModel, Field
from typing import List
from app.initialization import gemini_pro_model_langchain
import logging

logger = logging.getLogger(__name__)

class SubQueryResult(BaseModel):
        sub_query: str = Field(
            ..., description="A specific subquery to be transformed into a property graph query."
        )
        graph_query: str = Field(
            ..., description="The structured query generated for the property graph index."
        )
        
class QueryEngine:
    
    COMBINED_TEMPLATE = """You are an expert at converting user questions into database queries and queries for a vectorstore.
    You have access to a database of information about LLMs. 

    Your task involves two responsibilities:
    1. Perform query decomposition: Given a user question, break it down into distinct sub-questions that you need to 
    answer in order to address the original question. Ensure the sub-questions are clear and distinct.
    2. Convert queries for a vectorstore: For each sub-question, strip out information that is not relevant to the retrieval task and convert it into a query suitable for a vectorstore.

    Additional Instructions:
    - If there are acronyms or words you are not familiar with, do not try to rephrase them.
    - Ensure that each query or sub-question is actionable and specific for efficient retrieval.

    Here is the user query: {query}
    """

    PROPERTY_GRAPH_TEMPLATE = """You are an assistant tasked with transforming a natural language query into identified properties for a property graph index. 
    Your goal is to extract only the possible nodes (key entities) from the user's query. Do not include relationships, explanations, or additional context.

    Return the output in the following format:
    <node_1>, <node_2>, <node_3>, ...

    Example:
    If the user query is "How do LLMs work?", the output should be:
    LLMs, Working

    Here is the user query: {subquery}
    """

    def __init__(self):
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.COMBINED_TEMPLATE),
            ("human", "{query}"),
        ])
        self.parser = PydanticToolsParser(tools=[self.SubQuery])
        self.query_analyzer = self.prompt | gemini_pro_model_langchain.with_structured_output(self.SubQuery)

    class SubQuery(BaseModel):
        sub_queries: List[str] = Field(
            ...,
            description="A list of very specific subqueries against the database.",
        )

    def build_property_graph_chain(self, llm):
        prompt_template = PromptTemplate(
            template=self.PROPERTY_GRAPH_TEMPLATE,
            input_variables=["subquery"]
        )
        return prompt_template | llm

    def process_subqueries(self, llm, subqueries: List[str]) -> List[SubQueryResult]:
        logger.debug(f"Processing {len(subqueries)} subqueries")
        property_graph_chain = self.build_property_graph_chain(llm)
        
        results = []
        for subquery in subqueries:
            logger.debug(f"Processing subquery: {subquery}")
            graph_query_result = property_graph_chain.invoke({"subquery": subquery})
            results.append(SubQueryResult(sub_query=subquery, graph_query=graph_query_result.content))
        
        return results

    def process_query(self, query: str):
        logger.info(f"Processing query: {query}")
        subqueries = self.query_analyzer.invoke({"query": query})
        results = self.process_subqueries(gemini_pro_model_langchain, subqueries.sub_queries)
        
        return results