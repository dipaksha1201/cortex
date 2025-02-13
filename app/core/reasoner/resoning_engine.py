from typing import List
from app.core.interface.reasoning_classes import SubQueryContext , ReasoningStep, ThinkingOutput
from app.core.reasoner.query_engine import QueryEngine , SubQueryResult
from app.core.reasoner.retrievers.vector_retriever import VectorRetriever
from app.core.reasoner.retrievers.knowledge_graph_retriever import KnowledgeGraphRetriever, KnowledgeGraphRetrieverError
from .composers.composer import Composer
from .composers.thinking_composer import ThinkingComposer
import logging
from app.logging_config import reasoning_logger
from concurrent.futures import ThreadPoolExecutor
import asyncio
import nest_asyncio
nest_asyncio.apply()
logger = logging.getLogger(__name__)

class ReasoningEngineError(Exception):
    """Custom exception for ReasoningEngine errors"""
    pass

class ReasoningEngine:
    
    def __init__(self, username: str, query: str):
        if not username or not query:
            raise ValueError("Username and query must not be empty")
            
        self.username = username
        self.query = query
        reasoning_logger.info("Reasoning Engine initialized for project: %s", username)
        reasoning_logger.info("Query provided: %s", query)

    def generate_reasoning(self) -> str:
        try:
            query_engine = QueryEngine()
            subqueries = query_engine.process_query(self.query)
            if not subqueries:
                raise ReasoningEngineError("No subqueries generated from query")
            logger.info("Subqueries generated: %s", subqueries)
            return subqueries
        except Exception as e:
            logger.error(f"Error generating reasoning: {e}")
            raise ReasoningEngineError(f"Failed to generate reasoning: {str(e)}")

    def send_subqueries_to_retrievers(self, subqueries: List[SubQueryResult], index_name) -> List[SubQueryContext]:
        if not subqueries:
            raise ValueError("Subqueries list cannot be empty")
        if not index_name:
            raise ValueError("Index name cannot be empty")
            
        logger.info("Sending subqueries to retrievers for index: %s", index_name)
        
        try:
            vector_retriever = VectorRetriever(index_name=index_name)
            kg_retriever = KnowledgeGraphRetriever(index_name=index_name)
        except Exception as e:
            logger.error(f"Error initializing retrievers: {e}")
            raise ReasoningEngineError(f"Failed to initialize retrievers: {str(e)}")
        
        async def process_subquery(subquery):
            logger.info("Processing subquery: %s", subquery)
            try:
                # Run vector retrieval and KG retrieval concurrently
                vector_results = vector_retriever.retrieve(subquery.sub_query)
                kg_results = await kg_retriever.aretrieve(subquery.graph_query)
                
                return SubQueryContext(
                    subquery=subquery.sub_query,
                    graph_query=subquery.graph_query,
                    vector_context=vector_results,
                    knowledge_graph_context=kg_results
                )
            except KnowledgeGraphRetrieverError as e:
                logger.error(f"KG retriever error for subquery {subquery}: {e}")
                raise
            except Exception as e:
                logger.error(f"Error processing subquery: {e}")
                raise ReasoningEngineError(f"Failed to process subquery: {str(e)}")

        async def process_all_subqueries():
            tasks = [process_subquery(subquery) for subquery in subqueries]
            results = []
            try:
                # Process up to 4 subqueries at a time
                for i in range(0, len(tasks), 4):
                    batch = tasks[i:i+4]
                    batch_results = await asyncio.gather(*batch, return_exceptions=True)
                    for result in batch_results:
                        if isinstance(result, Exception):
                            raise result
                        results.append(result)
                return results
            except Exception as e:
                logger.error(f"Error processing subquery batch: {e}")
                raise ReasoningEngineError(f"Failed to process subquery batch: {str(e)}")

        try:
            loop = asyncio.get_event_loop()
            results = loop.run_until_complete(process_all_subqueries())
            if not results:
                logger.warning("No results retrieved from subqueries")
            logger.info("All subqueries processed with results: %s", results)
            return results
        except Exception as e:
            logger.error(f"Error in send_subqueries_to_retrievers: {e}")
            raise ReasoningEngineError(f"Failed to retrieve results: {str(e)}")
    
    def compose_subqueries(self, subqueries: List[SubQueryContext]) -> List[ReasoningStep]:
        if not subqueries:
            raise ValueError("Subqueries list cannot be empty")
            
        logger.info("Composing reasoning steps for original query: %s", self.query)
        
        try:
            composer = Composer()
        except Exception as e:
            logger.error(f"Error initializing composer: {e}")
            raise ReasoningEngineError(f"Failed to initialize composer: {str(e)}")
        
        def _process_subquery(subquery):
            try:
                logger.info("Composing reasoning step for subquery: %s", subquery)
                context = composer.get_context_from_subquery(subquery)
                reasoning_step = ReasoningStep(
                    query=subquery.subquery,
                    properties=subquery.graph_query,
                    context=context
                )
                logger.info("Reasoning step created: %s", reasoning_step)
                reasoning_logger.info("Reasoning step created: %s", reasoning_step)
                return reasoning_step
            except Exception as e:
                logger.error(f"Error processing subquery in composer: {e}")
                raise ReasoningEngineError(f"Failed to compose reasoning step: {str(e)}")

        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                reasoning_steps = list(executor.map(_process_subquery, subqueries))
            if not reasoning_steps:
                logger.warning("No reasoning steps composed")
            return reasoning_steps
        except Exception as e:
            logger.error(f"Error in compose_subqueries: {e}")
            raise ReasoningEngineError(f"Failed to compose reasoning steps: {str(e)}")
    
    def compose_answer(self, reasoning_steps: List[ReasoningStep]) -> str:
        if not reasoning_steps:
            raise ValueError("Reasoning steps list cannot be empty")
            
        logger.info("Composing final answer for original query: %s", self.query)
        try:
            composer = ThinkingComposer()
            final_answer = composer.think(self.query, reasoning_steps)
            if not final_answer:
                logger.warning("Empty final answer generated")
            return final_answer
        except Exception as e:
            logger.error(f"Error composing final answer: {e}")
            raise ReasoningEngineError(f"Failed to compose final answer: {str(e)}")
    
    def compose_table(self, final_answer: str) -> str:
        if not final_answer:
            raise ValueError("Final answer cannot be empty")
            
        logger.info("Composing table for final answer")
        try:
            composer = Composer()
            table = composer.get_table_from_output(final_answer)
            reasoning_logger.info("Table composed: %s", table)
            return table
        except Exception as e:
            logger.error(f"Error composing table: {e}")
            raise ReasoningEngineError(f"Failed to compose table: {str(e)}")
    
    def start_reasoning(self):
        logger.info("Starting process_reasoning for project: %s", self.username)
        try:
            subqueries = self.generate_reasoning()
            retrieved_results = self.send_subqueries_to_retrievers(subqueries, self.username)
            reasoning_steps = self.compose_subqueries(retrieved_results)
            final_answer = self.compose_answer(reasoning_steps)
            reasoning_logger.info("Final answer composed: %s", final_answer)
            output_table = self.compose_table(final_answer)
            logger.info("Process_reasoning completed for project: %s", self.username)
            return ThinkingOutput(reasoning=reasoning_steps, final_answer=final_answer, table=output_table)
        except (ValueError, ReasoningEngineError, KnowledgeGraphRetrieverError) as e:
            logger.error(f"Error in reasoning process: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in reasoning process: {e}")
            raise ReasoningEngineError(f"Unexpected error in reasoning process: {str(e)}")

