from pydantic import BaseModel
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
from langchain_core.tools import tool

class MemoryUpdate(BaseModel):
    updated_summary: str
    recall_memory: str
    title:str
    
class RecallMemory(BaseModel):
    memory: str
    summary: str
    ai_response: str

def search_memory(user_id:str ,query: str, top_k: int = 5, need_breif_recall_memory: bool = True) -> list[str]:
    """Search for memories in the database based on semantic similarity.

    Args:
        query (str): The search query.
        top_k (int): The number of results to return.

    Returns:
        list[str]: A list of relevant memories.
    """

    embeddings = utils.get_embeddings()
    vector = embeddings.embed_query(query)
    response = utils.get_index().query(
        vector=vector,
        filter={                                
            "user_id": {"$eq": user_id},
            constants.TYPE_KEY: {"$eq": "recall"},
        },
        namespace=settings.SETTINGS.pinecone_namespace,
        include_metadata=True,
        top_k=top_k,
    )
    memories = []
    if need_breif_recall_memory:
        if matches := response.get("matches"):
            memories = [m["metadata"][constants.PAYLOAD_KEY] for m in matches]
            return memories
    else:
        if matches := response.get("matches"):
            for match in matches:
                RecallMemory(memory=match["metadata"][constants.PAYLOAD_KEY], summary=match["metadata"]["summary"], ai_response=match["metadata"]["ai_response"])
            return memories

def load_memories(state: schemas.State, config: RunnableConfig) -> schemas.State:
    """Load core and recall memories for the current conversation.

    Args:
        state (schemas.State): The current state of the conversation.
        config (RunnableConfig): The runtime configuration for the agent.

    Returns:
        schemas.State: The updated state with loaded memories.
    """
    config = ensure_config()
    configurable = utils.ensure_configurable(config)
    user_id = configurable["user_id"]
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
    convo_str = get_buffer_string(state["messages"])
    convo_str = tokenizer.decode(tokenizer.encode(convo_str)[:2048])

    with get_executor_for_config(config) as executor:
        futures = [
            executor.submit(search_memory.invoke, user_id,convo_str),
        ]
        
        recall_memories = futures[0].result()
    return {
        "recall_memories": recall_memories,
    }



