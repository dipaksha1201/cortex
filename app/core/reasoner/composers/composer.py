from app.initialization import gemini_pro_model_langchain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from app.core.interface.reasoning_classes import SubQueryContext

class Composer:
    
    CONTEXT_COMPOSER_TEMPLATE = """
        You are an advanced AI assistant tasked with answering user queries comprehensively and accurately. 
        You have access to the following contexts:
        1. **Knowledge Graph Context**: Structured relational data that represents relationships between entities.
        2. **Vector Store Context**: Semantically relevant unstructured information extracted from datasets.

        Guidelines:
        - Understand the query's intent and identify key entities, relationships, or data points.
        - Leverage the Knowledge Graph Context to retrieve structured and factual information.
        - Consult the Vector Store Context to provide additional unstructured and semantic context.
        - Cross-reference data from all contexts to ensure accuracy and eliminate inconsistencies.
        - Deliver a detailed, clear, and complete response while citing data sources when possible.

        Steps:
        1. Parse the query to understand its core intent.
        2. Retrieve and combine information from all contexts.
        3. Provide an in-depth and structured answer to the query.

        Respond to the query as per these instructions.
    """
    
    CONTEXT_COMPOSER_INPUT_TEMPLATE =  """
        Original Query: {query}

        Knowledge Graph Context: {knowledge_graph}

        Vector Store Context: {vector_store_context}
        """
    
    def __init__(self):
        self.general_llm = gemini_pro_model_langchain
        
    def build_context_composer_prompt(self, context : SubQueryContext):
        # Combine the system and human templates into a chat prompt
        chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.CONTEXT_COMPOSER_TEMPLATE),
            HumanMessagePromptTemplate.from_template(self.CONTEXT_COMPOSER_INPUT_TEMPLATE)
        ])

        formatted_prompt = chat_prompt.format(
            query=context.subquery,
            knowledge_graph=context.knowledge_graph_context,
            vector_store_context=context.vector_context
        )

        return formatted_prompt

    def get_context_from_subquery(self, subquery_context: SubQueryContext):
        context_prompt = self.build_context_composer_prompt(subquery_context)
        context_response = self.general_llm.invoke(context_prompt)
        return context_response.content