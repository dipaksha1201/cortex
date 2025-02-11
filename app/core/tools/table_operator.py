from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from app.initialization import gemini_pro_model_langchain
from app.logging_config import reasoning_logger

# Define a few-shot example to demonstrate the update process.
few_shot_examples = """
Example 1:
Input Text: 
“Real-time Conversational Adaptation (RCA) enhances LLMs by dynamically adjusting context in multi-turn dialogues. The system continuously monitors conversation topics and recalibrates contextual parameters in real time, ensuring that the dialogue remains coherent even when users introduce abrupt topic shifts. RCA is evaluated using three key metrics: a dialogue coherence score that measures the model’s ability to integrate context across multiple turns; response latency, quantified in milliseconds, which tracks the speed of reply generation; and user satisfaction, collected through post-interaction surveys that assess overall conversational quality. This comprehensive evaluation aims to balance rapid response generation with maintaining high conversation coherence and user approval.”

Existing table:
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

User Instructions: “Additionally, update the evaluation to include scalability and resource efficiency. The new metric should assess the system’s ability to handle increased load and measure computational cost in GFLOPS under heavy usage scenarios.”

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
}},
{{
“Evaluation Category”: “Scalability & Resource Efficiency”,
“Metric / Description”: “Throughput and computational cost measured in GFLOPS”,
“Benchmark / Scenario”: “Heavy load scenarios with increased concurrent dialogues”,
“Performance Improvement / Result”: “Maintains system performance under load while optimizing resource usage”
}}
]


Example 2:
Input Text: 
“Context-Aware Summarization (CAS) leverages multi-source context to produce summaries that accurately capture key information while maintaining coherence and relevance. The system integrates various contextual signals—such as article metadata, user query focus, and visual content—to generate concise and informative summaries. CAS is evaluated using several metrics: a summary relevance score that measures how well the essential points are captured; a compression ratio that indicates the level of condensation relative to the original text; and an information retention rate that assesses how effectively critical content is preserved. These evaluations are benchmarked against traditional summarization models to highlight improvements in both conciseness and contextual richness.”

Existing Table:
[
{{
“Evaluation Category”: “Summary Relevance”,
“Metric / Description”: “Summary relevance score measuring key information capture”,
“Benchmark / Scenario”: “Comparison with traditional summarization models”,
“Performance Improvement / Result”: “Improved accuracy in capturing essential points”
}},
{{
“Evaluation Category”: “Compression Ratio”,
“Metric / Description”: “Compression ratio indicating summary length reduction”,
“Benchmark / Scenario”: “Document summarization tasks”,
“Performance Improvement / Result”: “Significant text reduction while maintaining relevance”
}},
{{
“Evaluation Category”: “Information Retention”,
“Metric / Description”: “Information retention rate assessing preservation of critical content”,
“Benchmark / Scenario”: “Content-rich documents”,
“Performance Improvement / Result”: “Higher retention of key details compared to baselines”
}}
]

User Instructions: Can you update the existing table by adding a new column named ‘Numerical Score’? Please assign appropriate numerical values to each evaluation metric (e.g., Summary Relevance, Compression Ratio, Information Retention, and Multi-modal Integration) to provide a more quantitative assessment of the system’s performance.

Final Table:
[
{{
“Evaluation Category”: “Summary Relevance”,
“Metric / Description”: “Summary relevance score measuring key information capture”,
“Benchmark / Scenario”: “Comparison with traditional summarization models”,
“Performance Improvement / Result”: “Improved accuracy in capturing essential points”,
“Numerical Score”: 87
}},
{{
“Evaluation Category”: “Compression Ratio”,
“Metric / Description”: “Compression ratio indicating summary length reduction”,
“Benchmark / Scenario”: “Document summarization tasks”,
“Performance Improvement / Result”: “Significant text reduction while maintaining relevance”,
“Numerical Score”: 0.33
}},
{{
“Evaluation Category”: “Information Retention”,
“Metric / Description”: “Information retention rate assessing preservation of critical content”,
“Benchmark / Scenario”: “Content-rich documents”,
“Performance Improvement / Result”: “Higher retention of key details compared to baselines”,
“Numerical Score”: 92
}},
{{
“Evaluation Category”: “Multi-modal Integration”,
“Metric / Description”: “Evaluation of integration of text and visual content in the summary”,
“Benchmark / Scenario”: “Multi-modal summarization tasks”,
“Performance Improvement / Result”: “Enhanced integration of visual context, boosting overall summary informativeness”,
“Numerical Score”: 81
}}
]
"""

# Create the prompt template with placeholders for current_table and user_message.
prompt_template = f"""
You are a data assistant. Your task is to update an existing table based on user instructions.
You will be provided with the following inputs:
1. A detailed Input Text that offers contextual information relevant to the data.
2. An Existing Table represented as a JSON array of objects.
3. A User Instruction detailing specific modifications to be made to the table (e.g., modifications, additions, deletions).

Your response must:
- Dynamically determine the keys based on the input, but ensure that all objects (rows) share the same keys.
- Output the updated table in valid JSON format.
- Maintain the same structure (i.e., the same keys across all objects) across all objects.

Few-shot examples:
{few_shot_examples}

Now, update the table based on the following inputs.

Input Text:
{{input_text}}

Existing table:
{{current_table}}

User instructions:
"{{user_message}}"

FInal table:
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# Use a generic JSON output parser to convert the LLM's output into a Python object
parser = JsonOutputParser()
    
class TableOperator:
    
    @staticmethod
    def update_table_data(input_text: str, current_table: list[dict], instructions: str):
        llm = gemini_pro_model_langchain
        # Format the prompt with the provided input text, current table, and user message
        formatted_prompt = prompt.format(
            input_text=input_text,
            current_table=current_table,
            user_message=instructions
        )
        
        # Generate the response using the LLM
        response = llm.invoke(formatted_prompt)
        
        # Parse the response to extract the updated table
        updated_table = parser.parse(response.content)
        reasoning_logger.info("Updated table: %s", updated_table)
        return updated_table
        