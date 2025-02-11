from pydantic import BaseModel
from typing import Any, List

class SubQueryContext:
    def __init__(self, subquery: str, graph_query: str, vector_context: Any, knowledge_graph_context: Any):
        self.subquery = subquery
        self.graph_query = graph_query
        self.vector_context = vector_context
        self.knowledge_graph_context = knowledge_graph_context

class ReasoningStep(BaseModel):
    query: str
    properties: str
    context: Any

class ThinkingOutput(BaseModel):
    reasoning: List[ReasoningStep]
    final_answer: str
    table: List[dict]
  