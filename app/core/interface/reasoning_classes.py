from typing import Any

class ThinkingOuput:
    def __init__(self, thoughts: str, thinking_output: str):
        self.thinking_output = thinking_output
        self.thoughts = thoughts

    def __str__(self):
        return self.thinking_output

    def __repr__(self):
        return self.thinking_output

    def get_thoughts(self):
        return self.thoughts

    def get_output(self):
        return self.thinking_output

class SubQueryContext:
    def __init__(self, subquery: str, graph_query: str, vector_context: Any, knowledge_graph_context: Any):
        self.subquery = subquery
        self.graph_query = graph_query
        self.vector_context = vector_context
        self.knowledge_graph_context = knowledge_graph_context

class ReasoningStep:
    def __init__(self, query: str, properties: str, context: Any):
        self.query = query
        self.properties = properties
        self.context = context
    
    def __str__(self):
        return f"Query: {self.query}, Properties: {self.properties}, Context: {self.context}"

    def __repr__(self):
        return self.__str__()
  