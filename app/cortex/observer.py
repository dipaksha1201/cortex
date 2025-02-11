from datetime import datetime, timezone
import json
import logging
import uuid
from .graph import save_recall_memory, store_core_memory
from app.config import gemini_pro_model_langchain
import app.cortex._constants as constants
import app.cortex._schemas as schemas
import app.cortex._settings as settings
from ._utils import utils
from app.cortex import memory_functions
from langchain_core.runnables.config import (
    RunnableConfig,
    ensure_config,
)
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from typing import Literal, Optional, Tuple
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate
from app.logging_config import memory_logger


_EMPTY_VEC = [0.0] * 768

@tool
def store_core_memory(memory: str, index: Optional[int] = None) -> str:
    """Store a core memory in the database.

    Args:
        memory (str): The memory to store.
        index (Optional[int]): The index at which to store the memory.

    Returns:
        str: A confirmation message.
    """
    config = ensure_config()
    configurable = utils.ensure_configurable(config)
    path, memories = memory_functions.fetch_core_memories(configurable["user_id"])
    if index is not None:
        if index < 0 or index >= len(memories):
            return "Error: Index out of bounds."
        memories[index] = memory
    else:
        memories.insert(0, memory)
    documents = [
        {
            "id": path,
            "values": _EMPTY_VEC,
            "metadata": {
                constants.PAYLOAD_KEY: json.dumps({"memories": memories}),
                constants.PATH_KEY: path,
                constants.TIMESTAMP_KEY: datetime.now(tz=timezone.utc),
                constants.TYPE_KEY: "recall",
                "user_id": configurable["user_id"],
            },
        }
    ]
    memory_logger.info(f"Saving core memory: {documents}")
    utils.get_index().upsert(
        vectors=documents,
        namespace=settings.SETTINGS.pinecone_namespace,
    )
    return "Memory stored."

@tool
async def save_recall_memory(memory: str) -> str:
    """Save a memory to the database for later semantic retrieval.

    Args:
        memory (str): The memory to be saved.

    Returns:
        str: The saved memory.
    """
    config = ensure_config()
    configurable = utils.ensure_configurable(config)
    embeddings = utils.get_embeddings()
    vector = await embeddings.aembed_query(memory)
    current_time = datetime.now(tz=timezone.utc)
    path = constants.INSERT_PATH.format(
        user_id=configurable["user_id"],
        event_id=str(uuid.uuid4()),
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
                "user_id": configurable["user_id"],
            },
        }
    ]
    memory_logger.info(f"Saving memory: {documents}")
    utils.get_index().upsert(
        vectors=documents,
        namespace=settings.SETTINGS.pinecone_namespace,
    )
    return memory

all_tools = [save_recall_memory, store_core_memory]
llm_model = gemini_pro_model_langchain

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant with advanced long-term memory"
            " capabilities. Powered by a stateless LLM, you must rely on"
            " external memory to store information between conversations."
            " Utilize the available memory tools to store core and recall memories."
            " important details that will help you better attend to the user's"
            " needs and understand their context.\n\n"
            "Memory Usage Guidelines:\n"
            "1. Actively use memory tools (save_core_memory, save_recall_memory)"
            " to build a comprehensive understanding of the user.\n"
            "2. Make informed suppositions and extrapolations based on stored"
            " memories.\n"
            "3. Regularly reflect on past interactions to identify patterns and"
            " preferences.\n"
            "4. Update your mental model of the user with each new piece of"
            " information.\n"
            "5. Cross-reference new information with existing memories for"
            " consistency.\n"
            "6. Prioritize storing emotional context and personal values"
            " alongside facts.\n"
            "7. Use memory to anticipate needs and tailor responses to the"
            " user's style.\n"
            "8. Recognize and acknowledge changes in the user's situation or"
            " perspectives over time.\n"
            "9. Leverage memories to provide personalized examples and"
            " analogies.\n"
            "10. Recall past challenges or successes to inform current"
            " problem-solving.\n\n"
            "## Core Memories\n"
            "Core memories are fundamental to understanding the user and are"
            " always available:\n{core_memories}\n\n"
            "## Recall Memories\n"
            "Recall memories are contextually retrieved based on the current"
            " conversation:\n{recall_memories}\n\n"
            "## Instructions\n"
            " Use tools to persist"
            " information you want to retain in the next conversation. If you"
            " do call tools, all text preceding the tool call is an internal"
            " message. Respond AFTER calling the tool, once you have"
            " confirmation that the tool completed successfully.\n\n"
            "Current system time: {current_time}\n\n",
        ),
        ("placeholder", "{messages}"),
    ]
)


async def agent(state: schemas.State) -> schemas.State:
    """Process the current state and generate a response using the LLM.

    Args:
        state (schemas.State): The current state of the conversation.
        config (RunnableConfig): The runtime configuration for the agent.

    Returns:
        schemas.State: The updated state with the agent's response.
    """
    llm = utils.get_llm(llm_model)
    
    bound = prompt | llm.bind_tools(all_tools)
    core_str = (
        "<core_memory>\n" + "\n".join(state["core_memories"]) + "\n</core_memory>"
    )
    recall_str = (
        "<recall_memory>\n" + "\n".join(state["recall_memories"]) + "\n</recall_memory>"
    )
    prediction = await bound.ainvoke(
        {
            "messages": state["messages"],
            "core_memories": core_str,
            "recall_memories": recall_str,
            "current_time": datetime.now(tz=timezone.utc).isoformat(),
        }
    )
    memory_logger.info(f"Agent response:\n {prediction}")
    return {
        "messages": prediction,
    }
    
def route_tools(state: schemas.State) -> Literal["tools", "__end__"]:
    """Determine whether to use tools or end the conversation based on the last message.

    Args:
        state (schemas.State): The current state of the conversation.

    Returns:
        Literal["tools", "__end__"]: The next step in the graph.
    """
    msg = state["messages"][-1]
    if msg.tool_calls:
        return "tools"
    return END

# Create the graph and add nodes
builder = StateGraph(schemas.State, schemas.GraphConfig)
builder.add_node("load_memories",memory_functions.load_memories)
builder.add_node(agent)
builder.add_node("tools", ToolNode(all_tools))

# Add edges to the graph
builder.add_edge(START, "load_memories")
builder.add_edge("load_memories", "agent")
builder.add_conditional_edges("agent", route_tools)
builder.add_edge("tools", "agent")

# Compile the graph
observer = builder.compile()
