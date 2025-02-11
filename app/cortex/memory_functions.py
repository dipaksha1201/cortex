import json
from typing import Tuple
import app.cortex._constants as constants
import app.cortex._schemas as schemas
import app.cortex._settings as settings
from ._utils import utils
from langchain_core.runnables.config import (
    RunnableConfig,
    ensure_config,
    get_executor_for_config,
)
import tiktoken
from langchain_core.messages.utils import get_buffer_string

def fetch_core_memories(user_id: str) -> Tuple[str, list[str]]:
    """Fetch core memories for a specific user.

    Args:
        user_id (str): The ID of the user.

    Returns:
        Tuple[str, list[str]]: The path and list of core memories.
    """
    path = constants.PATCH_PATH.format(user_id=user_id)
    response = utils.get_index().fetch(
        ids=[path], namespace=settings.SETTINGS.pinecone_namespace
    )
    memories = []
    if vectors := response.get("vectors"):
        document = vectors[path]
        payload = document["metadata"][constants.PAYLOAD_KEY]
        memories = json.loads(payload)["memories"]
    return path, memories

from langchain_core.tools import tool

@tool
def search_memory(query: str, top_k: int = 5) -> list[str]:
    """Search for memories in the database based on semantic similarity.

    Args:
        query (str): The search query.
        top_k (int): The number of results to return.

    Returns:
        list[str]: A list of relevant memories.
    """
    config = ensure_config()
    configurable = utils.ensure_configurable(config)
    embeddings = utils.get_embeddings()
    vector = embeddings.embed_query(query)
    response = utils.get_index().query(
        vector=vector,
        filter={
            "user_id": {"$eq": configurable["user_id"]},
            constants.TYPE_KEY: {"$eq": "recall"},
        },
        namespace=settings.SETTINGS.pinecone_namespace,
        include_metadata=True,
        top_k=top_k,
    )
    memories = []
    if matches := response.get("matches"):
        memories = [m["metadata"][constants.PAYLOAD_KEY] for m in matches]
    return memories


def load_memories(state: schemas.State, config: RunnableConfig) -> schemas.State:
    """Load core and recall memories for the current conversation.

    Args:
        state (schemas.State): The current state of the conversation.
        config (RunnableConfig): The runtime configuration for the agent.

    Returns:
        schemas.State: The updated state with loaded memories.
    """
    configurable = utils.ensure_configurable(config)
    user_id = configurable["user_id"]
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
    convo_str = get_buffer_string(state["messages"])
    convo_str = tokenizer.decode(tokenizer.encode(convo_str)[:2048])

    with get_executor_for_config(config) as executor:
        futures = [
            executor.submit(fetch_core_memories, user_id),
            executor.submit(search_memory.invoke, convo_str),
        ]
        _, core_memories = futures[0].result()
        recall_memories = futures[1].result()
    return {
        "core_memories": core_memories,
        "recall_memories": recall_memories,
    }



