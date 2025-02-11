from typing import List
from langchain_core.runnables.config import (
    ensure_config,
)
from app.data_layer.models.conversation import Message
from app.data_layer.services.conversation_service import ConversationService
from ._utils import utils
from app.core.tools.table_operator import TableOperator
from langchain.tools import Tool
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class TableOperatorInput(BaseModel):
    input_text: str = Field(..., description="The information required to update the table. The information should be extracted from the conversation and should be detailed and a long paragraph.")
    table_modification_instruction: str = Field(..., description="Clearly define the specific modifications the user wants to make refer the conversation if required and draft a formatted table modification instruction.")
    
# Define your external retrieval tool
@tool("TableOperator", parse_docstring=True)
def table_operator(input_text: str, table_modification_instruction:str) -> str:
    """
    Updates the table data based on the provided input text and modification instructions.

    Args:
        input_text (str): The information required to update the table. This should be detailed and extracted from the conversation. Refernt the message related to the table and provide the content attached to table in a long paragraph(400 - 500 words).
        table_modification_instruction (str): Clearly defined specific modifications the user wants to make, referring to the conversation if required.

    Returns:
        str: A message indicating the updated table.
    """
    # Your retrieval logic here (e.g., search a vectorstore or an API call)
    config = ensure_config()
    configurable = utils.ensure_configurable(config)
    conversation_service = ConversationService()
    current_conversation = conversation_service.get_conversation(configurable["thread_id"])
    table = TableOperator.update_table_data(input_text=current_conversation, current_table=current_conversation.output_table, instructions=table_modification_instruction)
    table_update_messsage = Message(
        sender="cortex",
        type="internal_knowledge",
        content="**Updated table**",
        table=table
    )
    conversation_service.store_message(
        message=table_update_messsage,
        user_id=configurable["user_id"],
        conversation_id=configurable["thread_id"]
    )
    return table_update_messsage