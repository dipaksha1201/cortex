from app.initialization import gemini_pro_model_langchain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from app.core.interface.reasoning_classes import SubQueryContext
from langchain_core.output_parsers import JsonOutputParser

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
    
    # Few-shot examples with an input text and its corresponding final table.
    FEW_SHOT_EXAMPLES_TABLE = """
    Example 1:
    Input Text:
    "KAG is also evaluated on retrieval effectiveness using Recall@k and via ablation studies, along with real-world applications in E-Government and E-Health, and enhanced LLM capabilities measured by MRR."
    Final Table:
    [
    {{
        "Evaluation Category": "Retrieval Effectiveness",
        "Metric / Description": "Recall@k for retrieving supporting facts",
        "Benchmark / Scenario": "Document retrieval tasks",
        "Performance Improvement / Result": "Higher Recall@k compared to other methods"
    }},
    {{
        "Evaluation Category": "Enhanced LLM Capabilities",
        "Metric / Description": "Improvements in natural language understanding and MRR",
        "Benchmark / Scenario": "Intrinsic LLM evaluation",
        "Performance Improvement / Result": "Higher accuracy and MRR"
    }}
    ]
    
    Example 2:
    Input Text:
    “Real-time Conversational Adaptation (RCA) enhances LLMs by dynamically adjusting context in multi-turn dialogues. The system continuously monitors conversation topics and recalibrates contextual parameters in real time, ensuring that the dialogue remains coherent even when users introduce abrupt topic shifts. RCA is evaluated using three key metrics: a dialogue coherence score that measures the model’s ability to integrate context across multiple turns; response latency, quantified in milliseconds, which tracks the speed of reply generation; and user satisfaction, collected through post-interaction surveys that assess overall conversational quality. This comprehensive evaluation aims to balance rapid response generation with maintaining high conversation coherence and user approval.”

    Final Table:
    [
    {{
    “Evaluation Category”: “Dialogue Coherence”,
    “Metric / Description”: “Dialogue coherence score measuring the ability to integrate context across multiple turns”,
    “Benchmark / Scenario”: “Multi-turn dialogues with abrupt topic shifts”,
    “Performance Improvement / Result”: “Ensures conversation remains coherent despite sudden topic changes”
    }},
    {{
    “Evaluation Category”: “Response Latency”,
    “Metric / Description”: “Response latency measured in milliseconds”,
    “Benchmark / Scenario”: “Real-time conversational interactions”,
    “Performance Improvement / Result”: “Tracks and minimizes reply generation delays”
    }},
    {{
    “Evaluation Category”: “User Satisfaction”,
    “Metric / Description”: “User satisfaction ratings from post-interaction surveys”,
    “Benchmark / Scenario”: “Evaluations of multi-turn dialogues”,
    “Performance Improvement / Result”: “Assesses overall conversational quality and approval”
    }}
    ]
    """

    # Create the prompt template with a placeholder {input_text} for the dynamic text.
    TABLE_COMPOSER_TEMPLATE = f"""
    You are an extraction assistant.
    Given an input text, extract all relevant entities and form a table represented as a JSON array of objects.
    Dynamically determine the keys based on the input, but ensure that all objects (rows) share the same keys.
    Do not include any extra text or markdown; output only valid JSON.

    Few-shot examples:
    {FEW_SHOT_EXAMPLES_TABLE}

    Now, given the input below, generate the table.
    Always validate the Final Table to ensure that all objects (rows) share the same keys.
    Input Text: 
    "{{input_text}}"
    
    Final Table:
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

    def build_table_composer_prompt(self, input_text: str):
        # Create the final prompt by inserting the input text into the template
        table_prompt =  ChatPromptTemplate.from_template(self.TABLE_COMPOSER_TEMPLATE)
        formatted_prompt = table_prompt.format(input_text=input_text)
        return formatted_prompt
    
    def get_table_from_output(self, input_text: str):
        table_prompt = self.build_table_composer_prompt(input_text)
        table_response = self.general_llm.invoke(table_prompt)
        parser = JsonOutputParser()
        final_table = parser.parse(table_response.content)
        return final_table
    
    def get_context_from_subquery(self, subquery_context: SubQueryContext):
        context_prompt = self.build_context_composer_prompt(subquery_context)
        context_response = self.general_llm.invoke(context_prompt)
        return context_response.content