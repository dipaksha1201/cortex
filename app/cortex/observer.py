from datetime import datetime, timezone
from app.initialization import gemini_flash_model_langchain
import app.cortex._constants as constants
import app.cortex._settings as settings
from ._utils import utils
from app.cortex import memory_functions

from langchain_core.prompts import ChatPromptTemplate
from app.logging_config import memory_logger
import tiktoken
from langchain_core.messages.utils import get_buffer_string
    
def save_recall_memory(memory: str, thread_id:str, user_id:str) -> str:
    """Save a memory to the database for later semantic retrieval.

    Args:
        memory (str): The memory to be saved.

    Returns:
        str: The saved memory.
    """
    embeddings = utils.get_embeddings()
    vector = embeddings.embed_query(memory)
    current_time = datetime.now(tz=timezone.utc)
    path = constants.INSERT_PATH.format(
        user_id=user_id,
        event_id=thread_id,
    )
    documents = [
        {
            "id": path,
            "values": vector,
            "metadata": {
                constants.PAYLOAD_KEY: memory,
                constants.PATH_KEY: path,
                constants.TIMESTAMP_KEY: current_time,
                constants.TYPE_KEY: "recall",
                "user_id": user_id,
            },
        }
    ]
    memory_logger.info(f"Saving memory: {documents}")
    utils.get_index().upsert(
        vectors=documents,
        namespace=settings.SETTINGS.pinecone_namespace,
    )
    return True

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant with advanced long-term memory"
            " capabilities. Powered by a stateless LLM, you must rely on"
            " external memory to store information between conversations."
            " Utilize the available memory tools to store summary and recall memories."
            " Important details that will help you better attend to the user's"
            " needs and understand their context.\n\n"
            "Memory Usage Guidelines:\n"
            "1. Actively use memory tools"
            " to build a comprehensive understanding of the user.\n"
            "2. Update your mental model of the user with each new piece of"
            " information.\n"
            "3. Cross-reference new information with existing memories for"
            " consistency.\n"
            "4. Prioritize storing emotional context and personal values"
            " alongside facts.\n"
            "5. Recognize and acknowledge changes in the user's situation or"
            " perspectives over time.\n"
            "6. Leverage memories to provide personalized examples and"
            " analogies. in the summary\n"
            "7. Regularly reflect on past interactions to identify patterns and"
            " preferences.\n"
            "8. Use memories to fill in gaps in your understanding and"
            " dont store information that is already in the memory to avoid duplication\n"

            "## Recall Memories\n"
            "Recall memories are contextually retrieved based on the current"
            " conversation:\n{recall_memories}\n\n"
            
            "## Instructions\n"
            "After processing the current conversation, generate the following outputs:\n"
            "1. **Updated Summary**: Provide a concise summary of the conversation so far or incrementally update an existing summary\n"
            "2. **Recall Memory**: Identify and list any new information that should be stored as recall memory.\n"
            "3. **Core Memory**: Identify and list any new information that should be stored as core memory.\n\n"
            "4. Give an appropriate title for the conversation\n\n"
            "Use these outputs to persist information you want to retain in the next conversation."
        ),
        ("placeholder", "{messages}"),
        ("human", "Current Summary: \n{current_summary}"),
    ]
)

def memory_builder(messages , current_summary, recall_memory):
    """Process the current state and generate a response using the LLM.

    Args:
        state (schemas.State): The current state of the conversation.
        config (RunnableConfig): The runtime configuration for the agent.

    Returns:
        schemas.State: The updated state with the agent's response.
    """
    recall_str = (
        "<recall_memory>\n" + "\n".join(recall_memory) + "\n</recall_memory>"
    )
    
    llm = gemini_flash_model_langchain.with_structured_output(memory_functions.MemoryUpdate)
    formatted_prompt = prompt.format(
        messages=messages,
        current_summary=current_summary,
        recall_memories=recall_str,
    )
    
    prediction = llm.invoke(formatted_prompt)
    memory_logger.info(f"Agent response:\n {prediction}")
    return prediction

def observer(messages, current_summary, user_id, thread_id):
    memory_logger.info(f"Starting observer for user_id: {user_id}, thread_id: {thread_id}")
    
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
    convo_str = get_buffer_string(messages)
    convo_str = tokenizer.decode(tokenizer.encode(convo_str)[:2048])
    memory_logger.debug(f"Processed conversation string length: {len(convo_str)}")
    
    recall_memories = memory_functions.search_memory(user_id, convo_str)
    memory_logger.info(f"Retrieved {len(recall_memories)} recall memories")
    
    memory_update = memory_builder(messages, current_summary, recall_memories)
    memory_logger.info("Memory builder completed")
    
    save_recall_memory(memory_update.recall_memory, thread_id, user_id)
    memory_logger.info("Recall memory saved")
        
    return memory_update.updated_summary , memory_update.title
    
    
    
