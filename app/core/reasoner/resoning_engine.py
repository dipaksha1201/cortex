from typing import List
from app.core.interface.reasoning_classes import SubQueryContext , ReasoningStep, ThinkingOutput
from app.core.reasoner.query_engine import QueryEngine , SubQueryResult
from app.core.reasoner.retrievers.vector_retriever import VectorRetriever
from app.core.reasoner.retrievers.knowledge_graph_retriever import KnowledgeGraphRetriever
from .composers.composer import Composer
from .composers.thinking_composer import ThinkingComposer
import logging
import json
from app.logging_config import reasoning_logger
logger = logging.getLogger(__name__)

# subqueries = [SubQueryResult(sub_query='What are the unique characteristics of Large Language Models (LLMs)?', graph_query='Large Language Models, Characteristics'), SubQueryResult(sub_query='What factors have contributed to the success of LLMs?', graph_query='Factors, Success, LLMs'), SubQueryResult(sub_query='Are the unique characteristics of LLMs the sole reason for their success?', graph_query='LLMs, Unique_characteristics, Success')]


class ReasoningEngine:
    
    def __init__(self, username: str, query: str):
        self.username = username
        self.query = query
        reasoning_logger.info("Reasoning Engine initialized for project: %s", username)
        reasoning_logger.info("Query provided: %s", query)

    def generate_reasoning(self) -> str:
        query_engine = QueryEngine()
        subqueries = query_engine.process_query(self.query)
        logger.info("Subqueries generated: %s", subqueries)
        return subqueries

    def send_subqueries_to_retrievers(self, subqueries: List[SubQueryResult], index_name) -> List[SubQueryContext]:
        logger.info("Sending subqueries to retrievers for index: %s", index_name)
        vector_retriever = VectorRetriever(index_name=index_name)
        kg_retriever = KnowledgeGraphRetriever(index_name=index_name)
        
        results = []
        for subquery in subqueries:
            logger.info("Processing subquery: %s", subquery)
            vector_results = vector_retriever.retrieve(subquery.sub_query)
            kg_results = kg_retriever.retrieve(subquery.graph_query)
            result_struct = SubQueryContext(
                subquery=subquery.sub_query,
                graph_query=subquery.graph_query,
                vector_context=vector_results,
                knowledge_graph_context=kg_results
            )
            results.append(result_struct)
        
        logger.info("All subqueries processed with results: %s", results)
        return results
    
    def compose_subqueries(self, subqueries: List[SubQueryContext]) -> List[ReasoningStep]:
        logger.info("Composing reasoning steps for original query: %s", self.query)
        composer = Composer()
        reasoning_steps = []
        for subquery in subqueries:
            logger.info("Composing reasoning step for subquery: %s", subquery)
            context = composer.get_context_from_subquery(subquery)
            reasoning_step = ReasoningStep(
            query=subquery.subquery,
            properties=subquery.graph_query,
            context=context
            )
            logger.info("Reasoning step created: %s", reasoning_step)
            reasoning_logger.info("Reasoning step created: %s", reasoning_step)
            reasoning_steps.append(reasoning_step)
        
        return reasoning_steps
    
    def compose_answer(self, reasoning_steps: List[ReasoningStep]) -> str:
        logger.info("Composing final answer for original query: %s", self.query)
        composer = ThinkingComposer()
        final_answer = composer.think(self.query ,reasoning_steps)
        return final_answer    
    
    def compose_table(self, final_answer: str) -> str:
        logger.info("Composing table for final answer: ")
        composer = Composer()
        table = composer.get_table_from_output(final_answer)
        reasoning_logger.info("Table composed: %s", table)
        return table
    
    def start_reasoning(self):
        logger.info("Starting process_reasoning for project: %s", self.username)
        subqueries = self.generate_reasoning()
        retrieved_results = self.send_subqueries_to_retrievers(subqueries, self.username)
        reasoning_steps = self.compose_subqueries(retrieved_results)
        final_answer = self.compose_answer(reasoning_steps)
        reasoning_logger.info("Final answer composed: %s", final_answer)
        output_table = self.compose_table(final_answer)
        logger.info("Process_reasoning completed for project: %s", self.username)
        return ThinkingOutput(reasoning=reasoning_steps, final_answer=final_answer, table=output_table)

