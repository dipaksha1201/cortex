from typing import List
from app.initialization import google_client, gemini_pro_model_langchain
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from app.core.interface.reasoning_classes import ReasoningStep 
import logging

class ThinkingComposer:
    
    QUERY_COMPOSER_TEMPLATE = """
        You have been provided with an original query and a series of subquery contexts.
        Each subquery includes a smaller query, some properties, and contextual information.

        Original Query:
        {original_query}

        Subquery Contexts:
        {formatted_reasoning_steps}

        Instructions:
        1. Think and plan about the original query and the subqueries.
        2. Combine and synthesize the insights from each subquery.
        3. Draft a conclusive answer that addresses the "Original Query" thoroughly.
        4. Be detailed, accurate, and informative in your response.
        5. Return a properly formatted markdown string as your final answer.

        Do not include any extra text or markdown;
        Final Answer:
        """

    def __init__(self):
        # self.thinking_client = google_client
        self.thinking_client = gemini_pro_model_langchain
        self.reasoning_instructions = PromptTemplate(
            input_variables=["original_query", "formatted_reasoning_steps"],
            template=self.QUERY_COMPOSER_TEMPLATE
        )
        
    def format_reasoning_steps(self, reasoning_steps: List[ReasoningStep]) -> str:
        """
        Utility to convert a list of ReasoningStep objects into a formatted string.
        """
        steps_str = []
        for i, step in enumerate(reasoning_steps, start=1):
            step_info = (
                f"Subquery order {i}:\n"
                f"Subquery: {step.query}\n"
                f"Properties: {step.properties}\n"
                f"Context: {step.context}\n"
            )
            steps_str.append(step_info)
        return "\n".join(steps_str)
    
    def generate_thinking_context(self, original_query: str, reasoning_steps: List[ReasoningStep]) -> str:
        """
        Use the PromptTemplate to build a final prompt string.
        """
        formatted_steps = self.format_reasoning_steps(reasoning_steps)
        return self.reasoning_instructions.format(
            original_query=original_query,
            formatted_reasoning_steps=formatted_steps
        )

    def generate_thinking_output(self, context: str):
        # config = {'thinking_config': {'include_thoughts': True}}
        # response = self.thinking_client.models.generate_content(
        #     model='gemini-2.0-flash-thinking-exp',
        #     contents=context,
        #     config=config
        # )
        response = self.thinking_client.invoke(context)
        return response
        
    def think(self, original_query: str, reasoning_steps: List[ReasoningStep]):
        logging.info("Generating thinking context for the original query.")
        context = self.generate_thinking_context(original_query, reasoning_steps)
        
        logging.info("Generated context: %s", context)
        logging.info("Generating thinking output.")
        output = self.generate_thinking_output(context)
        
        logging.info("Generated output: %s", output)
        # thoughts = output.candidates[0].content.parts[0].text
        # thinking_output = output.candidates[0].content.parts[0].text
        thinking_output = output.content
        
        # logging.info("Extracted thoughts: %s", thoughts)
        # logging.info("Extracted thinking output: %s", thinking_output)
        
        return thinking_output
        